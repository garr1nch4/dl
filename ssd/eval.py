import torch
from tqdm import tqdm
from pprint import PrettyPrinter

from utils import calculate_mAP


def evaluate(test_loader, model, label_map, rev_label_map, device):
    model.eval()

    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, predicted_scores,
                min_score=0.01, max_overlap=0.45,
                top_k=200)

            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        APs, mAP = calculate_mAP(label_map, rev_label_map, device, det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    pp = PrettyPrinter()
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)









