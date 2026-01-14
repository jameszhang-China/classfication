from libs.engine.model import Classification

# model = Classification(model='cfg/resnet18.yaml', hyper=False)
# model.train()

# model = Classification(model='runs/default/last.pt', hyper=False)
# model.train()

# model = Classification(model='runs/default/last.pt', hyper=False)
# model.predict()

# model = Classification(model='runs/default/last.pt', hyper=False)
# model.val()

model = Classification(model='runs/default/last.pt', hyper=False)
model.export()