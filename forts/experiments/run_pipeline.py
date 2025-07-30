import logging
import sys
import traceback

import pandas as pd

from forts.data_pipeline.data_pipeline_setup import (
    DataPipeline,
    build_mixed_trainval,
    get_data_pipeline,
)
from forts.experiments.helper import (
    check_results_exist,
    cmd_parser,
    extract_frequency,
    extract_horizon,
    get_model_list,
    set_device,
)
from forts.gcs_utils import gcs_write_csv, get_gcs_path
from forts.metrics.evaluation_pipeline import evaluation_pipeline_forts_forecast
from forts.model_pipeline.model_pipeline import ModelPipeline, ModelPipelineCoreset

DATASET_GROUP_FREQ = {
    "Tourism": {
        "Monthly": {"FREQ": "ME", "H": 24},
    },
    "M1": {
        "Monthly": {"FREQ": "ME", "H": 24},
        "Quarterly": {"FREQ": "Q", "H": 8},
    },
    "M3": {
        "Monthly": {"FREQ": "ME", "H": 24},
        "Quarterly": {"FREQ": "Q", "H": 8},
        "Yearly": {"FREQ": "Y", "H": 4},
    },
    "M4": {
        "Monthly": {"FREQ": "ME", "H": 24},
        "Quarterly": {"FREQ": "Q", "H": 8},
    },
    "Traffic": {
        "Daily": {"FREQ": "D", "H": 30},
    },
    "M5": {
        "Daily": {"FREQ": "D", "H": 30},
    },
}

SOURCE_DATASET_GROUP_FREQ_TRANSFER_LEARNING = {
    "Tourism": {
        "Monthly": {"FREQ": "ME", "H": 24},
    },
    "M1": {
        "Monthly": {"FREQ": "ME", "H": 24},
        "Quarterly": {"FREQ": "Q", "H": 8},
    },
    "M3": {
        "Monthly": {"FREQ": "ME", "H": 24},
        "Quarterly": {"FREQ": "Q", "H": 8},
        "Yearly": {"FREQ": "Y", "H": 4},
    },
    "M4": {
        "Monthly": {"FREQ": "ME", "H": 24},
        "Quarterly": {"FREQ": "Q", "H": 8},
    },
    "Traffic": {
        "Daily": {"FREQ": "D", "H": 30},
    },
    "M5": {
        "Daily": {"FREQ": "D", "H": 30},
    },
}


def main():
    """Main function to run the experiments."""
    try:
        args = cmd_parser()
        set_device(use_gpu=args.use_gpu)

        results = []
        model_list = get_model_list()

        if args.coreset:
            LOO_RESULTS = []

            all_data_pipelines = {}
            for ds, groups in DATASET_GROUP_FREQ.items():
                for grp, meta in groups.items():
                    all_data_pipelines[(ds, grp)] = DataPipeline(
                        dataset_name=ds,
                        dataset_group=grp,
                        freq=meta["FREQ"],
                        horizon=meta["H"],
                        window_size=meta["H"],
                    )

            # leave-one-out
            for (
                target_ds,
                target_grp,
            ), target_data_pipeline in all_data_pipelines.items():
                all_models_exist = True
                for model_name, _ in model_list:
                    dataset_source = "MIXED"
                    dataset_group = f"ALL_BUT_{target_ds}_{target_grp}"
                    exists, _ = check_results_exist(
                        dataset=target_ds,
                        dataset_group=target_grp,
                        model_name=model_name,
                        horizon=target_data_pipeline.h,
                        mode="out_domain",
                        finetune=args.finetune,
                        dataset_source=dataset_source,
                        dataset_group_source=dataset_group,
                    )
                    if not exists:
                        all_models_exist = False
                        break
                if all_models_exist:
                    print(
                        f"[SKIP] All results for {target_ds}/{target_grp} from CORESET exist."
                    )
                    continue

                # gather the source pipelines except the held-out one
                source_pipelines = [
                    data_pipeline
                    for ds, data_pipeline in all_data_pipelines.items()
                    if ds != (target_ds, target_grp)
                ]

                dataset_source = "MIXED"
                dataset_group = f"ALL_BUT_{target_ds}_{target_grp}"
                mixed_trainval = build_mixed_trainval(
                    source_pipelines,
                    dataset_source=dataset_source,
                    dataset_group=dataset_group,
                )

                mixed_freq = "mixed"
                mixed_h = target_data_pipeline.h

                mixed_mp = ModelPipelineCoreset(
                    mixed_trainval,
                    freq=mixed_freq,
                    h=mixed_h,
                )
                mixed_mp.hyper_tune_and_train(
                    max_evals=20,
                    mode="out_domain_coreset",
                    dataset_source=dataset_source,
                    dataset_group_source=dataset_group,
                )

                heldout_mp = ModelPipeline(target_data_pipeline)

                for model_name, nf_model in mixed_mp.models.items():
                    row = {}

                    evaluation_pipeline_forts_forecast(
                        dataset=target_ds,
                        dataset_group=target_grp,
                        pipeline=heldout_mp,
                        model=nf_model,
                        horizon=target_data_pipeline.h,
                        freq=target_data_pipeline.freq,
                        period=target_data_pipeline.period,
                        row_forecast=row,
                        dataset_source=dataset_source,
                        dataset_group_source=dataset_group,
                        mode="out_domain",
                        window_size=target_data_pipeline.h,
                        window_size_source=target_data_pipeline.h,
                        finetune=args.finetune,
                    )
                    LOO_RESULTS.append(row)
            results.extend(LOO_RESULTS)
        else:
            for DATASET, SUBGROUPS in DATASET_GROUP_FREQ.items():
                for subgroup in SUBGROUPS.items():
                    dataset_group_results = []

                    FREQ = extract_frequency(subgroup)
                    H = extract_horizon(subgroup)
                    DATASET_GROUP = subgroup[0]

                    print(
                        f"Dataset: {DATASET}, Dataset-group: {DATASET_GROUP}, Frequency: {FREQ}"
                    )

                    if args.transfer_learning:
                        for (
                            DATASET_TL,
                            SUBGROUPS_TL,
                        ) in SOURCE_DATASET_GROUP_FREQ_TRANSFER_LEARNING.items():
                            for subgroup_tl in SUBGROUPS_TL.items():
                                all_models_exist = True
                                for model_name, _ in model_list:
                                    H_TL = extract_horizon(subgroup_tl)
                                    DATASET_GROUP_TL = subgroup_tl[0]

                                    exists, _ = check_results_exist(
                                        dataset=DATASET,
                                        dataset_group=DATASET_GROUP,
                                        model_name=model_name,
                                        horizon=H,
                                        mode="out_domain",
                                        finetune=args.finetune,
                                        dataset_source=DATASET_TL,
                                        dataset_group_source=DATASET_GROUP_TL,
                                    )
                                    if not exists:
                                        all_models_exist = False
                                        break

                                if all_models_exist:
                                    print(
                                        f"[SKIP] All results for {DATASET}/{DATASET_GROUP} from {DATASET_TL}/{DATASET_GROUP_TL} exist."
                                    )
                                    continue

                                data_pipeline_transfer_learning = get_data_pipeline(
                                    DATASET_TL, subgroup_tl
                                )
                                model_pipeline_transfer_learning = ModelPipeline(
                                    data_pipeline=data_pipeline_transfer_learning
                                )
                                model_pipeline_transfer_learning.hyper_tune_and_train(
                                    max_evals=20,
                                    mode="out_domain",
                                    dataset_source=DATASET_TL,
                                    dataset_group_source=DATASET_GROUP_TL,
                                )
                                data_pipeline = get_data_pipeline(DATASET, subgroup)
                                model_pipeline = ModelPipeline(
                                    data_pipeline=data_pipeline
                                )

                                for (
                                    model_name,
                                    model,
                                ) in model_pipeline_transfer_learning.models.items():
                                    row_forecast_tl = {}
                                    evaluation_pipeline_forts_forecast(
                                        dataset=DATASET,
                                        dataset_group=DATASET_GROUP,
                                        model=model,
                                        pipeline=model_pipeline,
                                        horizon=H,
                                        freq=FREQ,
                                        period=data_pipeline.period,
                                        row_forecast=row_forecast_tl,
                                        dataset_source=DATASET_TL,
                                        dataset_group_source=DATASET_GROUP_TL,
                                        mode="out_domain",
                                        window_size=H,
                                        window_size_source=H_TL,
                                        finetune=args.finetune,
                                    )
                                    dataset_group_results.append(row_forecast_tl)
                                    results.append(row_forecast_tl)
                    elif args.basic_forecasting:
                        all_models_exist = True
                        for model_name, _ in model_list:
                            exists, _ = check_results_exist(
                                dataset=DATASET,
                                dataset_group=DATASET_GROUP,
                                model_name=model_name,
                                horizon=H,
                                mode="basic_forecasting",
                                finetune=args.finetune,
                                dataset_source=DATASET,
                                dataset_group_source=DATASET_GROUP,
                            )
                            if not exists:
                                all_models_exist = False
                                break
                        if all_models_exist:
                            print(
                                f"[SKIP] All results for basic forecasting on {DATASET}/{DATASET_GROUP} exist."
                            )
                            continue

                        data_pipeline = get_data_pipeline(DATASET, subgroup)
                        model_pipeline = ModelPipeline(data_pipeline=data_pipeline)
                        model_pipeline.hyper_tune_and_train(
                            max_evals=20,
                            mode="basic_forecasting",
                            dataset_source=DATASET,
                            dataset_group_source=DATASET_GROUP,
                        )

                        for model_name, model in model_pipeline.models.items():
                            row_forecast = {}

                            evaluation_pipeline_forts_forecast(
                                dataset=DATASET,
                                dataset_group=DATASET_GROUP,
                                model=model,
                                pipeline=model_pipeline,
                                period=data_pipeline.period,
                                horizon=H,
                                freq=FREQ,
                                row_forecast=row_forecast,
                                window_size=H,
                                window_size_source=H,
                                mode="basic_forecasting",
                            )

                            dataset_group_results.append(row_forecast)
                            results.append(row_forecast)
                    else:
                        all_models_exist = True
                        for model_name, _ in model_list:
                            exists, _ = check_results_exist(
                                dataset=DATASET,
                                dataset_group=DATASET_GROUP,
                                model_name=model_name,
                                horizon=H,
                                mode="in_domain",
                                finetune=args.finetune,
                                dataset_source=DATASET,
                                dataset_group_source=DATASET_GROUP,
                            )
                            if not exists:
                                all_models_exist = False
                                break
                        if all_models_exist:
                            print(
                                f"[SKIP] All results for in-domain on {DATASET}/{DATASET_GROUP} exist."
                            )
                            continue
                        data_pipeline = get_data_pipeline(DATASET, subgroup)
                        model_pipeline = ModelPipeline(data_pipeline=data_pipeline)
                        model_pipeline.hyper_tune_and_train(
                            max_evals=20,
                            mode="in_domain",
                            dataset_source=DATASET,
                            dataset_group_source=DATASET_GROUP,
                        )

                        for model_name, model in model_pipeline.models.items():
                            row_forecast = {}

                            evaluation_pipeline_forts_forecast(
                                dataset=DATASET,
                                dataset_group=DATASET_GROUP,
                                model=model,
                                pipeline=model_pipeline,
                                period=data_pipeline.period,
                                horizon=H,
                                freq=FREQ,
                                row_forecast=row_forecast,
                                window_size=H,
                                window_size_source=H,
                                mode="in_domain",
                            )

                            dataset_group_results.append(row_forecast)
                            results.append(row_forecast)

        df_results = pd.DataFrame(results)

        results_path = get_gcs_path("results_forecast")
        final_results_path = f"{results_path}/final_results.csv"
        gcs_write_csv(df_results, final_results_path)

        print(f"Final forecast results saved to {final_results_path}")

    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
