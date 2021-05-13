#https://torchmetrics.readthedocs.io/en/stable/pages/overview.html

#echar un vistazo https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226
#https://pytorch-lightning.medium.com/torchmetrics-pytorch-metrics-built-to-scale-7091b1bec919

from torchmetrics import MetricCollection, Accuracy
def get_metrics_collections_base( prefix):
    
    metrics = MetricCollection(
            {
                "Accuracy":Accuracy(),
                "Top_3":Accuracy(top_k=3),
                "Top_5" :Accuracy(top_k=5),

            },
            prefix=prefix
            )
    
    
    return metrics
