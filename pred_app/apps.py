from django.apps import AppConfig
from .utils.model_utils import load_resnet50_model, load_swin_transformer_model, load_EffNetB6_model


class PredAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'pred_app'

    # Load the trained models
    resnet50 = load_resnet50_model()
    swin_transformer = load_swin_transformer_model()
    effNetB6 = load_EffNetB6_model()
