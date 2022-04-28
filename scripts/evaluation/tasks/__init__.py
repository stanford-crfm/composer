from .pubmed_qa_task_evaluator import PubMedQATaskEvaluator
from .med_qa_task_evaluator import MedQATaskEvaluator
from .ner_ebm_pico_task_evaluator import NEREBMPICOTaskEvaluator
from .ner_bc5cdr_task_evaluator import NERBC5CDRTaskEvaluator
from .covid_dialog_task_evaluator import CovidDialogTaskEvaluator
from .medparasimp_task_evaluator import MedParaSimpTaskEvaluator
from .meqsum_task_evaluator import MeQSumTaskEvaluator

all_evaluators = [
    PubMedQATaskEvaluator,
    MedQATaskEvaluator,
    NEREBMPICOTaskEvaluator,
    NERBC5CDRTaskEvaluator,
    CovidDialogTaskEvaluator,
    MedParaSimpTaskEvaluator,
    MeQSumTaskEvaluator,
]
