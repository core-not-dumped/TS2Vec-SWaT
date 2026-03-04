import wandb

class WandBLogger:
    def __init__(self):
        self.step = 0

    def log_val(self, threshold, attack_detection_rate, false_positive_rate, top_attack_percentage, top_normal_percentage, loss_ep):
        wandb.log({
            "threshold": threshold,
            "attack_detection_rate": attack_detection_rate,
            "false_positive_rate": false_positive_rate,
            "top_attack_percentage": top_attack_percentage,
            "top_normal_percentage": top_normal_percentage,
            "loss_ep": loss_ep,
        }, step=self.step)
        self.step += 1