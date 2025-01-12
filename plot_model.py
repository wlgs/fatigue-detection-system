from modules.FatigueEvaluator import FatigueEvaluator

if __name__ == '__main__':
    evaluator = FatigueEvaluator()
    evaluator.setup_bayesian_network()
    model = evaluator.get_model()
    model_daft = model.to_daft()
    model_daft.render()
    model_daft.savefig("model.png")
