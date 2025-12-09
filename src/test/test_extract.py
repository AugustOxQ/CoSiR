from src.utils.tools import precompute_coco_embeddings, visualize_angular_semantics_fast
from src.model.cosirmodel import CoSiRModel
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from src.dataset.cosir_datamodule import CoSiRValidationDataset
import torch
from src.eval import EvaluationManager, EvaluationConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_extract():
    model = CoSiRModel(label_dim=2).to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    coco_test_dataset = CoSiRValidationDataset(
        annotation_path="/data/SSD/coco/annotations/coco_karpathy_test.json",
        image_path="/data/SSD/coco/images",
        processor=processor,
        ratio=1.0,
    )
    test_loader = DataLoader(
        coco_test_dataset, batch_size=64, shuffle=False, num_workers=8
    )

    evaluation_config = EvaluationConfig(
        device=device,  # type: ignore
        k_vals=[1, 5, 10],
        train_max_batches=25,
        print_metrics=True,
        evaluation_interval=5,
    )
    evaluator = EvaluationManager(evaluation_config)

    representatives = torch.randn(5, 2).to(device)
    labels_all = torch.randn(1000, 2).to(device)

    test_results_detailed = evaluator.evaluate_test(
        model=model,
        processor=processor,
        dataloader=test_loader,
        label_embeddings=representatives,  # Your label embeddings
        epoch=0,
        return_detailed_results=True,
    )

    (
        all_img_emb,
        all_txt_emb,
        all_raw_text,
        _,
        image_to_text_map,
        _,
    ) = test_results_detailed  # type: ignore

    fig = visualize_angular_semantics_fast(
        labels_all,
        model,
        (all_img_emb, all_txt_emb, all_raw_text, image_to_text_map),
        save_path="angular_semantics_fast.png",
        device=device,
    )


def main():
    test_extract()


if __name__ == "__main__":
    main()
