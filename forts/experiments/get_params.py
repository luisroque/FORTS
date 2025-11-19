import pandas as pd
from neuralforecast.auto import (
    AutoiTransformer,
    AutoKAN,
    AutoNHITS,
    AutoPatchTST,
    AutoTFT,
    AutoTSMixer,
    AutoxLSTM,
    AutoTimeMixer,
    AutoNBEATS
)

from forts.gcs_utils import gcs_list_files, gcs_write_csv, get_gcs_path
from forts.model_pipeline.core.core_extension import CustomNeuralForecast

BASE_DIR = get_gcs_path("model_weights_out_domain/hypertuning")
OUTPUT_CSV = "model_parameter_counts.csv"
results_out_dir = get_gcs_path("results_forecast_out_domain_summary")

MODEL_CLASSES = {
    ("AutoNHITS", AutoNHITS),
    ("AutoTFT", AutoTFT),
    ("AutoPatchTST", AutoPatchTST),
    ("AutoTSMixer", AutoTSMixer),
    ("AutoiTransformer", AutoiTransformer),
    ("AutoKAN", AutoKAN),
    ("AutoxLSTM", AutoxLSTM),
    ("AutoTimeMixer", AutoTimeMixer),
    ("AutoNBEATS", AutoNBEATS),
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    records = []

    all_files = gcs_list_files(BASE_DIR, extension="_neuralforecast")

    for model_path in all_files:
        filename = model_path.split("/")[-1]
        if filename.startswith("MIXED"):
            for name, ModelClass in MODEL_CLASSES:
                if name not in filename:
                    continue

                try:
                    # dummy init for loading
                    try:
                        auto_model = ModelClass(h=1, num_samples=1, config={})
                    except Exception:
                        auto_model = ModelClass(
                            h=1, num_samples=1, config={}, n_series=1
                        )

                    nf = CustomNeuralForecast(models=[auto_model], freq="D")
                    nf = nf.load(path=model_path)

                    torch_model = nf.models[0]
                    n_params = count_parameters(torch_model)

                    records.append(
                        {
                            "model_name": name,
                            "path": model_path,
                            "num_parameters": n_params,
                        }
                    )

                    print(f"{name} — {n_params:,} parameters")

                except Exception as e:
                    print(f"Failed to load {model_path}: {e}")

    df = pd.DataFrame(records)
    df_agg = df.groupby("model_name", as_index=False)["num_parameters"].mean()

    csv_path = f"{results_out_dir}/{OUTPUT_CSV}"
    gcs_write_csv(df_agg, csv_path)

    print(f"\nSaved model parameter counts to '{csv_path}'.")


if __name__ == "__main__":
    main()
