from . import unet

model_from_name = {}


model_from_name["unet_mini"] = unet.unet_mini
model_from_name["unet"] = unet.unet
model_from_name["vgg_unet"] = unet.vgg_unet
model_from_name["resnet50_unet"] = unet.resnet50_unet
model_from_name["mobilenet_unet"] = unet.mobilenet_unet
