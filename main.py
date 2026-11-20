import matplotlib.pyplot as plt
import numpy as np

models = ["ResNet18", "ResNet50", "MobileNetV2", "EfficientNet"]
accuracy = [93.78, 96.60, 91.70, 91.70]
auc      = [98.55, 99.69, 98.10, 98.10]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, accuracy, width, label="Accuracy (%)", color="#1F9D8F")
ax.bar(x + width/2, auc, width, label="AUC (%)", color="#102A43")
ax.set_ylim(85, 101)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Score (%)")
ax.set_title("Model Comparison on processed_cropped1_test")
ax.legend()
plt.tight_layout()
plt.savefig("fig_c8_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()