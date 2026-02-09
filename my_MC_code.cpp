#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>    

using namespace std;

int main() {
    // Domain: [-1, 1] x [-1, 1]
    double x_min = -1.0, x_max = 1.0;

    // Sample sizes: n = 10^power for power in {2,...,8}
    vector<int> sample_sizes_power = {2,3,4,5,6,7,8};
    int num_repeats = 10; // Repeat the experiment 10 times for each sample size

    vector<double> pi_estimate(sample_sizes_power.size());
    vector<double> percent_error(sample_sizes_power.size());
    vector<double> runtime(sample_sizes_power.size());

    // Keep the sample sizes for CSV output
    vector<size_t> sample_sizes(sample_sizes_power.size());

    // Random number generation setup
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dist(x_min, x_max);

    for (size_t j = 0; j < sample_sizes_power.size(); ++j) {
        size_t n = pow(10, sample_sizes_power[j]);
        sample_sizes[j] = n; // Save for CSV

        // Store results for this sample size over num_repeats
        vector<double> pi_est_runs(num_repeats);
        vector<double> percent_error_runs(num_repeats);
        vector<double> runtime_runs(num_repeats);

        for (int repeat = 0; repeat < num_repeats; ++repeat) {
            int in_circle = 0;
            auto t0 = chrono::high_resolution_clock::now();

            // Generate n points and count those in the unit circle
            for (size_t i = 0; i < n; ++i) {
                double x = dist(gen);
                double y = dist(gen);
                if (x*x + y*y <= 1.0) {
                    ++in_circle;
                }
            }

            auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = t1 - t0;
            double pi_est = 4.0 * in_circle / n;
            double pct_error = 100.0 * (M_PI - pi_est) / M_PI;

            // Store for this repeat
            pi_est_runs[repeat] = pi_est;
            percent_error_runs[repeat] = pct_error;
            runtime_runs[repeat] = elapsed.count();
        }

        // Average over num_repeats
        double pi_sum = 0.0, error_sum = 0.0, time_sum = 0.0;
        for (int repeat = 0; repeat < num_repeats; ++repeat) {
            pi_sum += pi_est_runs[repeat];
            error_sum += percent_error_runs[repeat];
            time_sum += runtime_runs[repeat];
        }
        pi_estimate[j] = pi_sum / num_repeats;
        percent_error[j] = error_sum / num_repeats;
        runtime[j] = time_sum / num_repeats;
    }

    // --- CSV output ---
    ofstream csv("serial_pi_results.csv");
    csv << "SampleSize,PiEstimate,PercentError,RuntimeSeconds\n";
    for (size_t j = 0; j < sample_sizes_power.size(); ++j) {
        csv << sample_sizes[j] << ","
            << pi_estimate[j] << ","
            << percent_error[j] << ","
            << runtime[j] << "\n";
    }
    csv.close();
    cout << "Results saved to serial_pi_results.csv\n";

    return 0;
}