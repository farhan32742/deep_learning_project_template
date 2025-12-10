from Deep_learning_projects.config.configuration import ConfigurationManager
from Deep_learning_projects.components.model_evalution_mlflow import Evaluation
from Deep_learning_projects.utils import log

STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(config=eval_config)
        
        # This function now does two things:
        # 1. Calculates the YOLO metrics (mAP, Precision, Recall)
        # 2. Automatically saves them to 'scores.json'
        evaluation.evaluation()
        
        # Optional: Uncomment this to log to DagsHub/MLflow
        # Ensure you have set your env variables (MLFLOW_TRACKING_URI, USERNAME, PASSWORD)
        # evaluation.log_into_mlflow()



if __name__ == '__main__':
    try:
        log.info(f"*******************")
        log.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        log.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        log.exception(e)
        raise e