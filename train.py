from libs.engine.model import Classification

# model = Classification(model='cfg/resnet18.yaml', hyper=False)
# model.train()

model = Classification(model='best.pt', hyper=False)
model.predict()

