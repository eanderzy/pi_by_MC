# estimating pi using monte carlo sampling 
# steps:
# define domain: unit square
# place points in domain
# count the points that have d = x*x + y*y <= 1 
# (ie in the unit circle)
# fraction of points in circle over in square is pi/4
# --> multiply the fraction of points in circle vs in square by 4
# then compute % error for each sample size
# also time each sample size
# sample sizes (10^n) where n = 2,3,4,5,6,7,8

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tabulate import tabulate

def monte_carlo_pi(sample_sizes, num_repeats=10):
    """
    Runs Monte Carlo estimation of pi for the given sample_sizes. 
    Returns df with columns: sample_size, pi_estimate, percent_error, runtime.
    """
    actual_sample_sizes = [10**n for n in sample_sizes]
    pi_est = np.empty(len(sample_sizes), dtype=float)
    pct_err = np.empty(len(sample_sizes), dtype=float)
    run_times = np.empty(len(sample_sizes), dtype=float)

    for j, n in enumerate(actual_sample_sizes):
        pi_est_runs = np.empty(num_repeats, dtype=float)
        pct_err_runs = np.empty(num_repeats, dtype=float)
        run_time_runs = np.empty(num_repeats, dtype=float)

        for repeat in range(num_repeats):
            t0 = time.perf_counter()
            # Vectorized sampling for speed
            points = np.random.uniform(-1, 1, size=(n, 2))
            distances_squared = np.sum(points**2, axis=1)
            in_circle = np.count_nonzero(distances_squared <= 1)
            t1 = time.perf_counter()

            pi_est_runs[repeat] = 4 * in_circle / n
            pct_err_runs[repeat] = 100 * ((np.pi - pi_est_runs[repeat]) / np.pi)
            run_time_runs[repeat] = t1 - t0

        pi_est[j] = np.mean(pi_est_runs)
        pct_err[j] = np.mean(pct_err_runs)
        run_times[j] = np.mean(run_time_runs)

    df = pd.DataFrame({
        'sample_size': actual_sample_sizes,
        'pi_est': pi_est,
        'pct_err': pct_err,
        'run': run_times
    })

    return df

# UM GPT
""" write a function that saves figures in a publication style/quality. Include a legend, axis labels, but no titles. Save as both PNG and PDF"""
def publication_plot(x1, y1, style1, label1,  x2, y2, style2, label2, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(x1, y1, style1, label=label1, 
            linewidth=2.0, markersize=6)
    ax.plot(x2, y2, style2, label=label2, 
            linewidth=2.0, markersize=6)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', which='major', 
                    labelsize=10, direction='in',
                    length=6, width=1)
    ax.legend(loc='best', fontsize=11, frameon=False)
    ax.grid(which='major', linestyle=':', linewidth=0.5, alpha=0.4)
    
    # Format x ticks as powers of ten, LaTeX labels
    ax.set_xticks(x1)
    ax.set_xticklabels([r'$10^{{{}}}$'.format(n) for n in x1], fontsize=10)
    plt.tight_layout()
    fig.savefig(f'{filename}.png', dpi=600, bbox_inches='tight')
    fig.savefig(f'{filename}.pdf', bbox_inches='tight')
    plt.close(fig)

# UM GPT
"""Write a function that take my variables and formats them in a latex table"""
def latex_comparison_table(sample_sizes, pi_est, pct_err, run_times, 
                           cpp_sample, cpp_pi, cpp_pct, cpp_run):
    merged_rows = []
    for i in range(len(sample_sizes)):
        merged_rows.append([
            f"10$^{sample_sizes[i]}$", # Sample Size (formatted for LaTeX)
            f"{pi_est[i]:.5f}",  # Pi Estimate Python
            f"{cpp_pi[i]:.5f}",  # Pi Estimate C++
            f"{pct_err[i]:.4f}",  # % Error Python
            f"{cpp_pct[i]:.4f}",  # % Error C++
            f"{run_times[i]:.3f}",  # Time (s) Python
            f"{cpp_run[i]:.3f}",  # Time (s) C++
        ])

    headers = [
        "Sample Size", 
        "$pi$ Estimate python", 
        "$pi$ Estimate C++", 
        "% Error python", 
        "% Error C++", 
        "Time (s) python", 
        "Time (s) C++"
    ]

    latex_table = tabulate(merged_rows, headers, tablefmt="latex")

    print(latex_table)

if __name__ == "__main__":
    sample_sizes = [2,3,4,5,6,7,8]
    num_repeats = 10

    #df = monte_carlo_pi(sample_sizes, num_repeats)
    #df.to_csv('python_pi_monte_carlo_results.csv', index=False)

    df = pd.read_csv('python_pi_monte_carlo_results.csv')

    # Read C++ results for comparison/plotting
    cpp_data = pd.read_csv('CPP_pi_monte_carlo_results.csv')
    cpp_sample = cpp_data['sample_size']
    cpp_pi = cpp_data['pi_estimate']
    cpp_pct = cpp_data['percent_error']
    cpp_run = cpp_data['runtime']

    # Display comparison latex table for publication
    latex_comparison_table(
        sample_sizes=sample_sizes,
        pi_est=df['pi_estimate'].values,
        pct_err=df['percent_error'].values,
        run_times=df['runtime'].values,
        cpp_sample=cpp_sample.values,
        cpp_pi=cpp_pi.values,
        cpp_pct=cpp_pct.values,
        cpp_run=cpp_run.values
    )

    python_sample_size = df['sample_size'].values  
    cpp_sample_size = cpp_sample.values

    publication_plot(
        python_sample_size, df['pi_estimate'].values, 'k--', 'Python',
        cpp_sample_size, cpp_pi.values, 'b-', 'C++',
        'Sample size', 'Average $\pi$ estimate', 'pi_estimate_pub'
    )
    publication_plot(
        python_sample_size, df['percent_error'].values, 'k--', 'Python',
        cpp_sample_size, cpp_pct.values, 'b-', 'C++',
        'Sample size', 'Average % error', 'pct_error_pub'
    )
    publication_plot(
        python_sample_size, df['runtime'].values, 'k--', 'Python',
        cpp_sample_size, cpp_run.values, 'b-', 'C++',
        'Sample size', 'Average run time (s)', 'runtime_pub'
    )