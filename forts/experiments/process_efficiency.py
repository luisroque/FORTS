import pandas as pd

from forts.gcs_utils import gcs_list_files, gcs_read_csv, get_gcs_path

base_path = get_gcs_path("model_weights_out_domain/hypertuning_final")
plots_dir = get_gcs_path("plots")
results_out_dir = get_gcs_path("results_forecast_out_domain_summary")


performance_results = [
    f.split("/")[-1]
    for f in gcs_list_files(base_path, extension=".csv")
    if not f.startswith("MIXED")
]

results_combined = []
for result in performance_results:
    csv_path = f"{base_path}/{result}"
    df = gcs_read_csv(csv_path)

    df["Dataset"] = result.split("_")[0]
    df["Group"] = result.split("_")[1]
    df["Method"] = result.split("_")[2]
    df = df[["Dataset", "Group", "Method", "time_total_s", "loss"]]

    results_combined.append(df)

idx = (
    pd.concat(results_combined, ignore_index=True)
    .groupby(["Dataset", "Group", "Method"])["loss"]
    .idxmin()
)

min_loss_df = pd.concat(results_combined, ignore_index=True).loc[idx]


results_df = min_loss_df.groupby("Method")["time_total_s"].sum().reset_index()

method_order = sorted(results_df["Method"].unique().tolist())

csv_path = f"{results_out_dir}/training_time_stats_per_method.csv"
results_df.to_csv(csv_path, index=False)
