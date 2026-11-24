import argparse
from pathlib import Path
from spam_classifier.config import AppConfig, DataConfig, TrainConfig, PathConfig, load_config
from spam_classifier.data import load_raw_data, log_dataset_statistics,preprocess_and_save
from spam_classifier.model import build_pipeline,eval_and_export,save_model
from sklearn.model_selection import train_test_split, GridSearchCV
from spam_classifier.logger import get_logger

logger = get_logger(__name__)

def run_training(cfg: AppConfig):
    df = load_raw_data(cfg.paths.raw_data, cfg.data)
    log_dataset_statistics(df,cfg.data)
    processed_df = preprocess_and_save(df,cfg.paths.processed_data,cfg.data)

    x = processed_df[cfg.data.text_column]
    y = processed_df[cfg.data.label_column]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=cfg.train.test_size,random_state=cfg.train.random_state, stratify=y)
    logger.info("Dataset size: Train %d , Test %d",len(x_train), len(x_test))
    pipeline = build_pipeline(cfg.train)

    logger.info("Start of Grid Search")

    cv = 5

    grid_search = GridSearchCV(estimator=pipeline, param_grid={"mnb__alpha":cfg.train.alpha},cv=cv, scoring="f1",n_jobs=-1)
    pipeline = grid_search.fit(x_train,y_train)

    logger.info("The optimal alpha value selected is: %s",grid_search.best_params_.get("mnb__alpha"))
    eval_and_export(pipeline=pipeline,x_test=x_test,y_test=y_test,metrics_path=cfg.paths.metrics,cm_path=cfg.paths.confusion_matrix, error_analysis_path=cfg.paths.error_analysis)
    save_model(pipeline,cfg.paths.model)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the spam classifier")
    parser.add_argument("-c","--config",type=Path,default=Path("config/config.yaml"),help="Path to YAML configuration file")

    args = parser.parse_args()
    cfg = load_config(args.config)
    run_training(cfg)

if __name__ == "__main__":
    main()