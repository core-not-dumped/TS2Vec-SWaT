import wandb

class WandBLogger:
    def __init__(self):
        self.step = 0

    def log_val(self, threshold, attack_detection_rate, false_positive_rate):
        wandb.log({
            "threshold": threshold,
            "attack_detection_rate": attack_detection_rate,
            "false_positive_rate": false_positive_rate,
        }, step=self.step)
        self.step += 1