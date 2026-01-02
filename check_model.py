from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=100, ignore_mismatched_sizes=True)
print(model.vit.encoder.layer[0])
 