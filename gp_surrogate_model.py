"""
Gaussian Process Surrogate Model for Microchannel Heat Transfer
================================================================
Predicts local Nusselt number profiles in turbulent rectangular microchannels.
Trained on CFD data, validated against the published correlation.

Reference:
    Rahaman et al. (2026), "Turbulent Flow in a Rectangular Microchannel...",
    ASME Journal of Heat and Mass Transfer. DOI: 10.1115/1.4070848

Author: Harshavardhan Ramachandran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =============================================================================
# PUBLISHED CORRELATION
# =============================================================================

def gnielinski_Nu(Re, Pr):
    """
    Gnielinski correlation for fully-developed turbulent flow.
    Used as Nu_fd in the published local Nu correlation.
    """
    f = (0.790 * np.log(Re) - 1.64)**(-2)
    Nu = (f/8) * (Re - 1000) * Pr / (1 + 12.7 * np.sqrt(f/8) * (Pr**(2/3) - 1))
    return Nu


def published_correlation(x_mm, Dh_microns, alpha, Re, Pr):
    """
    Published correlation for local Nusselt number (Eq. 17 from paper).
    
    Nu_x / Nu_fd = 3.4996 × (X/Dh)^(-0.0314) × α^(-0.0294) × Re^(-0.1182) × Pr^(0.0393)
    
    where Nu_fd is from the Gnielinski correlation.
    """
    X_over_Dh = (x_mm * 1000) / Dh_microns
    Nu_fd = gnielinski_Nu(Re, Pr)
    ratio = 3.4996 * (X_over_Dh**(-0.0314)) * (alpha**(-0.0294)) * (Re**(-0.1182)) * (Pr**0.0393)
    return Nu_fd * ratio


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath):
    """Load training data from Excel file."""
    df = pd.read_excel(filepath, sheet_name='GP_Input_Data')
    
    # Compute X/Dh for correlation
    df['X_over_Dh'] = (df['x_mm'] * 1000) / df['Dh_microns']
    
    print(f"Loaded {len(df)} data points from {df['Case'].nunique()} cases")
    print(f"Parameter ranges:")
    print(f"  Dh:    {df['Dh_microns'].min():.0f} – {df['Dh_microns'].max():.0f} μm")
    print(f"  α:     {df['alpha'].min():.2f} – {df['alpha'].max():.2f}")
    print(f"  Re:    {df['Re'].min():.0f} – {df['Re'].max():.0f}")
    print(f"  Pr:    {df['Pr'].min():.2f} – {df['Pr'].max():.2f}")
    print(f"  x*:    {df['x_star'].min():.6f} – {df['x_star'].max():.6f}")
    
    return df


def split_by_case(df, test_cases):
    """Split data by case ID for proper validation."""
    train = df[~df['Case'].isin(test_cases)].copy()
    test = df[df['Case'].isin(test_cases)].copy()
    return train, test


# =============================================================================
# GAUSSIAN PROCESS MODEL
# =============================================================================

def create_gp_model():
    """
    Create GP with Matérn kernel.
    
    - Matérn (ν=2.5): physical data is smooth but not infinitely differentiable
    - WhiteKernel: accounts for noise/discretization error in CFD data
    """
    kernel = (
        ConstantKernel(100.0, (1e-2, 1e4)) *
        Matern(length_scale=[1.0]*5, length_scale_bounds=(1e-3, 1e3), nu=2.5) +
        WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42
    )


def train_gp(X_train, y_train):
    """Train GP and return model with fitted scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    gp = create_gp_model()
    gp.fit(X_scaled, y_train)
    
    return gp, scaler


def predict_gp(gp, scaler, X):
    """Predict with uncertainty."""
    X_scaled = scaler.transform(X)
    y_pred, y_std = gp.predict(X_scaled, return_std=True)
    return y_pred, y_std


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cross_validation(df, n_splits=5):
    """
    Grouped k-fold cross-validation.
    Entire cases are held out to test generalization to unseen geometries.
    """
    # Filter to well-characterized region for evaluation
    df_eval = df[df['x_star'] <= 0.002].copy()
    
    feature_cols = ['Dh_microns', 'alpha', 'Re', 'Pr', 'x_star']
    X = df_eval[feature_cols].values
    y = df_eval['Nu'].values
    groups = df_eval['Case'].values
    
    gkf = GroupKFold(n_splits=n_splits)
    results = {'r2': [], 'mape': [], 'mae': []}
    
    print(f"\n{'='*60}")
    print(f"Running {n_splits}-Fold Grouped Cross-Validation")
    print(f"{'='*60}")
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        gp, scaler = train_gp(X_train, y_train)
        y_pred, _ = predict_gp(gp, scaler, X_test)
        
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mae = mean_absolute_error(y_test, y_pred)
        
        results['r2'].append(r2)
        results['mape'].append(mape)
        results['mae'].append(mae)
        
        test_cases = np.unique(groups[test_idx])
        print(f"Fold {fold+1}: Test cases {list(test_cases)[:4]}... | R²={r2:.4f}, MAPE={mape:.1f}%")
    
    print(f"\nCross-Validation Summary:")
    print(f"  R²:   {np.mean(results['r2']):.4f} ± {np.std(results['r2']):.4f}")
    print(f"  MAPE: {np.mean(results['mape']):.1f}% ± {np.std(results['mape']):.1f}%")
    
    return results


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_on_test_set(df, test_cases):
    """Train final model and evaluate against published correlation."""
    feature_cols = ['Dh_microns', 'alpha', 'Re', 'Pr', 'x_star']
    
    # Split by case
    df_train, df_test = split_by_case(df, test_cases)
    
    # Train on ALL training data points
    X_train = df_train[feature_cols].values
    y_train = df_train['Nu'].values
    
    print(f"\nTraining on {len(df_train)} points from {df_train['Case'].nunique()} cases...")
    gp, scaler = train_gp(X_train, y_train)
    print(f"Optimized kernel:\n{gp.kernel_}")
    
    # Evaluate only on x* <= 0.002 region (well-characterized)
    df_test_eval = df_test[df_test['x_star'] <= 0.002].copy()
    X_test = df_test_eval[feature_cols].values
    y_test = df_test_eval['Nu'].values
    
    print(f"Evaluating on {len(df_test_eval)} points from {df_test_eval['Case'].nunique()} test cases")
    
    # GP predictions
    y_pred_gp, y_std_gp = predict_gp(gp, scaler, X_test)
    
    # Published correlation predictions
    y_pred_corr = df_test_eval.apply(
        lambda r: published_correlation(r['x_mm'], r['Dh_microns'], r['alpha'], r['Re'], r['Pr']),
        axis=1
    ).values
    
    # Store predictions
    df_test_eval = df_test_eval.copy()
    df_test_eval['Nu_pred_gp'] = y_pred_gp
    df_test_eval['Nu_std_gp'] = y_std_gp
    df_test_eval['Nu_pred_corr'] = y_pred_corr
    
    # Compute metrics
    metrics = {
        'GP Surrogate': {
            'r2': r2_score(y_test, y_pred_gp),
            'mape': np.mean(np.abs((y_test - y_pred_gp) / y_test)) * 100,
            'mae': mean_absolute_error(y_test, y_pred_gp)
        },
        'Published Correlation': {
            'r2': r2_score(y_test, y_pred_corr),
            'mape': np.mean(np.abs((y_test - y_pred_corr) / y_test)) * 100,
            'mae': mean_absolute_error(y_test, y_pred_corr)
        }
    }
    
    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'R²':>10} {'MAPE':>10} {'MAE':>10}")
    print('-'*55)
    for method, m in metrics.items():
        print(f"{method:<25} {m['r2']:>10.4f} {m['mape']:>9.1f}% {m['mae']:>10.2f}")
    
    improvement = (metrics['Published Correlation']['mape'] - metrics['GP Surrogate']['mape']) / metrics['Published Correlation']['mape'] * 100
    print(f"\nGP reduces error by {improvement:.0f}% compared to published correlation.")
    
    return gp, scaler, df_test_eval, metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_profile_comparisons(df_test, output_dir):
    """Plot Nu vs x* for each test case."""
    test_cases = sorted(df_test['Case'].unique())
    n_cases = len(test_cases)
    n_cols = 2
    n_rows = (n_cases + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    axes = axes.flatten()
    
    for i, case_num in enumerate(test_cases):
        ax = axes[i]
        case_data = df_test[df_test['Case'] == case_num].sort_values('x_star')
        
        x = case_data['x_star'].values
        Nu_actual = case_data['Nu'].values
        Nu_gp = case_data['Nu_pred_gp'].values
        Nu_std = case_data['Nu_std_gp'].values
        Nu_corr = case_data['Nu_pred_corr'].values
        
        Dh = case_data['Dh_microns'].iloc[0]
        alpha = case_data['alpha'].iloc[0]
        Re = case_data['Re'].iloc[0]
        Pr = case_data['Pr'].iloc[0]
        
        # Plot
        ax.scatter(x, Nu_actual, c='black', s=60, zorder=5, label='CFD Data', edgecolors='white')
        ax.plot(x, Nu_gp, 'b-', linewidth=2, label='GP Surrogate')
        ax.fill_between(x, Nu_gp - 2*Nu_std, Nu_gp + 2*Nu_std, alpha=0.25, color='blue', label='95% CI')
        ax.plot(x, Nu_corr, 'r--', linewidth=2, label='Published Correlation')
        
        ax.set_xlabel('x*')
        ax.set_ylabel('Nu')
        ax.set_title(f'Case {case_num}: Dh={Dh:.0f}μm, α={alpha:.2f}, Re={Re:.0f}, Pr={Pr:.1f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # MAPE annotation
        mape_gp = np.mean(np.abs((Nu_actual - Nu_gp) / Nu_actual)) * 100
        mape_corr = np.mean(np.abs((Nu_actual - Nu_corr) / Nu_actual)) * 100
        ax.text(0.98, 0.98, f'GP: {mape_gp:.1f}%\nCorr: {mape_corr:.1f}%',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Nu(x*) Profiles: GP Surrogate vs Published Correlation', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'profile_comparisons.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/profile_comparisons.png")


def plot_parity(df_test, output_dir):
    """Parity plot comparing GP and correlation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    y_actual = df_test['Nu'].values
    lims = [y_actual.min()-5, y_actual.max()+5]
    
    plot_data = [
        ('Nu_pred_gp', 'GP Surrogate', 'blue'),
        ('Nu_pred_corr', 'Published Correlation', 'red')
    ]
    
    for ax, (col, name, color) in zip(axes, plot_data):
        y_pred = df_test[col].values
        r2 = r2_score(y_actual, y_pred)
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        
        ax.scatter(y_actual, y_pred, alpha=0.6, s=40, c=color, edgecolors='white')
        ax.plot(lims, lims, 'k--', linewidth=2, label='Perfect')
        ax.plot(lims, [l*1.15 for l in lims], 'gray', ls=':', lw=1.5)
        ax.plot(lims, [l*0.85 for l in lims], 'gray', ls=':', lw=1.5, label='±15%')
        
        ax.set_xlabel('Actual Nu (CFD)')
        ax.set_ylabel(f'Predicted Nu')
        ax.set_title(f'{name}\nR² = {r2:.4f}, MAPE = {mape:.1f}%')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parity_plots.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/parity_plots.png")


def plot_error_distribution(df_test, output_dir):
    """Compare error distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    y_actual = df_test['Nu'].values
    err_gp = (df_test['Nu_pred_gp'].values - y_actual) / y_actual * 100
    err_corr = (df_test['Nu_pred_corr'].values - y_actual) / y_actual * 100
    
    # Histogram
    ax = axes[0]
    bins = np.linspace(-25, 25, 26)
    ax.hist(err_gp, bins=bins, alpha=0.6, label=f'GP (std={np.std(err_gp):.1f}%)', color='blue')
    ax.hist(err_corr, bins=bins, alpha=0.6, label=f'Correlation (std={np.std(err_corr):.1f}%)', color='red')
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.axvline(-15, color='gray', linestyle=':', linewidth=1.5)
    ax.axvline(15, color='gray', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Relative Error (%)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()
    
    # Box plot
    ax = axes[1]
    bp = ax.boxplot([err_gp, err_corr], labels=['GP Surrogate', 'Published\nCorrelation'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax.axhline(0, color='black', linestyle='--', linewidth=2)
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Error Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/error_distribution.png")


def plot_new_geometry(gp, scaler, output_dir):
    """Predict for a new geometry not in training data."""
    Dh, alpha, Re, Pr = 550, 1.25, 7500, 8.0
    x_star = np.linspace(0.0001, 0.0018, 50)  # Stay within validated range
    
    # GP prediction
    X_new = np.column_stack([np.full(50, Dh), np.full(50, alpha),
                             np.full(50, Re), np.full(50, Pr), x_star])
    Nu_gp, Nu_std = predict_gp(gp, scaler, X_new)
    
    # Correlation prediction
    x_mm = x_star * Dh * Re * Pr / 1000
    Nu_corr = [published_correlation(x, Dh, alpha, Re, Pr) for x in x_mm]
    Nu_gn = gnielinski_Nu(Re, Pr)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_star, Nu_gp, 'b-', linewidth=2.5, label='GP Surrogate')
    ax.fill_between(x_star, Nu_gp - 2*Nu_std, Nu_gp + 2*Nu_std, 
                    alpha=0.25, color='blue', label='95% CI')
    ax.plot(x_star, Nu_corr, 'r--', linewidth=2, label='Published Correlation')
    ax.axhline(Nu_gn, color='gray', linestyle='-.', label=f'Gnielinski Nu_fd = {Nu_gn:.1f}')
    
    ax.set_xlabel('x*')
    ax.set_ylabel('Local Nusselt Number')
    ax.set_title(f'Prediction for New Geometry: Dh={Dh}μm, α={alpha}, Re={Re}, Pr={Pr}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.text(0.02, 0.02, f'GP uncertainty: ±{Nu_std.mean():.1f} (avg)\nThis geometry is NOT in training data',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'new_geometry_prediction.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/new_geometry_prediction.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("GP SURROGATE MODEL FOR MICROCHANNEL HEAT TRANSFER")
    print("="*60)
    
    # Load all data
    df = load_data('GP_Training_Data.xlsx')
    
    # Cross-validation
    cv_results = run_cross_validation(df, n_splits=5)
    
    # Final train/test split
    unique_cases = df['Case'].unique()
    np.random.shuffle(unique_cases)
    test_cases = unique_cases[:8]
    
    print(f"\n{'='*60}")
    print(f"FINAL MODEL")
    print(f"{'='*60}")
    print(f"Test cases: {sorted(test_cases)}")
    
    # Train and evaluate
    gp, scaler, df_test, metrics = evaluate_on_test_set(df, test_cases)
    
    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    plot_profile_comparisons(df_test, output_dir)
    plot_parity(df_test, output_dir)
    plot_error_distribution(df_test, output_dir)
    plot_new_geometry(gp, scaler, output_dir)
    
    # Save results
    df_test.to_excel(os.path.join(output_dir, 'test_predictions.xlsx'), index=False)
    print(f"Saved: {output_dir}/test_predictions.xlsx")
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
