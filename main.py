import torch

from utils.visualize_results import plot_combined_metric_results, plot_performance_with_error_bars
from utils.analysis_data import run_analysis_pipeline
from data.data_preprocessing import get_features, preprocessing_data


if __name__ == "__main__":
    # --- GPU Setup ---
    run_gpu = 0 # GPU ID for MI calculation, -1 for CPU
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and run_gpu >= 0:
        try:
            device = torch.device(f'cuda:{run_gpu}')
            print(f"Attempting to use device: {device}\t{torch.cuda.get_device_name(run_gpu)}")
            _ = torch.tensor([1.0, 2.0]).to(device)
            print(f"Successfully using device: {device}")
        except Exception as e:
            print(f"Error initializing CUDA device {run_gpu}: {e}\nFalling back to CPU.")
            device = torch.device("cpu")
    else:
        if run_gpu >=0 and not torch.cuda.is_available():
            print("CUDA not available, MI calculation will use CPU.")
        device = torch.device("cpu")
        print(f"Using device for MI calculation: {device}")


    target_columns, excluded_features, patient_report_features, sleep_related_features = get_features()
    combined_df_imputed, final_target_names = preprocessing_data()

    # 1. Define Feature Sets for each Model
    features_for_model_A = [col for col in combined_df_imputed.columns if col not in final_target_names and col not in excluded_features]
    features_for_model_B = [col for col in features_for_model_A if col not in patient_report_features]
    features_for_model_C = [col for col in features_for_model_B if col not in sleep_related_features]

    # 2. Analysis pipeline each model
    results_a, mi_scores_a = run_analysis_pipeline(combined_df_imputed, features_for_model_A, final_target_names, device, "Full_Model")
    results_b, mi_scores_b = run_analysis_pipeline(combined_df_imputed, features_for_model_B, final_target_names, device, "Clinical_Sleep_Model")
    results_c, mi_scores_c = run_analysis_pipeline(combined_df_imputed, features_for_model_C, final_target_names, device, "Clinical_Only_Model")

    plot_combined_metric_results(results_a, results_b, results_c, metric_name="MSE")
    plot_combined_metric_results(results_a, results_b, results_c, metric_name="RMSE")

    plot_performance_with_error_bars(results_a, results_b, results_c, metric_name="MSE", target="CMI")
    plot_performance_with_error_bars(results_a, results_b, results_c, metric_name="MSE", target="VAS")

    plot_performance_with_error_bars(results_a, results_b, results_c, metric_name="RMSE", target="CMI")
    plot_performance_with_error_bars(results_a, results_b, results_c, metric_name="RMSE", target="VAS")
