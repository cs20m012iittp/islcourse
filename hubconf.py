
import torch
from torch import nn

def kali():
  print ('kali')
  
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
class cs20m012(nn.Module):
  def __init__(self, numChannels, classes):
    super(LeNet, self).__init__()
    # initialize first set of CONV => RELU => POOL layers
    self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
      kernel_size=(5, 5))
    self.relu1 = ReLU()
    self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    # initialize second set of CONV => RELU => POOL layers
    self.conv2 = Conv2d(in_channels=20, out_channels=50,
      kernel_size=(5, 5))
    self.relu2 = ReLU()
    self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    # initialize first (and only) set of FC => RELU layers
    self.fc1 = Linear(in_features=800, out_features=500)
    self.relu3 = ReLU()
    # initialize our softmax classifier
    self.fc2 = Linear(in_features=500, out_features=classes)
    self.logSoftmax = LogSoftmax(dim=1)

    
  def forward(self, x):
    # pass the input through our first set of CONV => RELU =>
    # POOL layers
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.maxpool1(x)
    # pass the output from the previous layer through the second
    # set of CONV => RELU => POOL layers
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.maxpool2(x)
    # flatten the output from the previous layer and pass it
    # through our only set of FC => RELU layers
    x = flatten(x, 1)
    x = self.fc1(x)
    x = self.relu3(x)
    # pass the output to our softmax classifier to get our output
    # predictions
    x = self.fc2(x)
    output = self.logSoftmax(x)
    # return the output predictions
    return output
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=5):
  model = None
  INIT_LR = 1e-3
  BATCH_SIZE = 64
  EPOCHS = 5
  # define the train and val splits
  TRAIN_SPLIT = 0.75
  VAL_SPLIT = 1 - TRAIN_SPLIT
  # set the device we will be using to train the model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("[INFO] loading the KMNIST dataset...")
  trainData = KMNIST(root="data", train=True, download=True,
	transform=ToTensor())
  testData = KMNIST(root="data", train=False, download=True,
	transform=ToTensor())
  # calculate the train/validation split
  print("[INFO] generating the train/validation split...")
  numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
  numValSamples = int(len(trainData) * VAL_SPLIT)
  (trainData, valData) = random_split(trainData,
	[numTrainSamples, numValSamples],
	generator=torch.Generator().manual_seed(42))
  
  # initialize the train, validation, and test data loaders
  trainDataLoader = DataLoader(trainData, shuffle=True,
	batch_size=BATCH_SIZE)
  valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
  testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)
  # calculate steps per epoch for training and validation set
  trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
  valSteps = len(valDataLoader.dataset) // BATCH_SIZE
  print("[INFO] initializing the LeNet model...")
  model = cs20m012(
	numChannels=1,
	classes=len(trainData.dataset.classes)).to(device)
  # initialize our optimizer and loss function
  opt = Adam(model.parameters(), lr=INIT_LR)
  lossFn = nn.NLLLoss()
  # initialize a dictionary to store training history
  H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
  }
  # measure how long training is going to take
  print("[INFO] training the network...")
  startTime = time.time()


 
  # loop over our epochs
  for e in range(0, EPOCHS):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0
    # loop over the training set
    for (x, y) in trainDataLoader:
    # send the input to the device
      (x, y) = (x.to(device), y.to(device))
    # perform a forward pass and calculate the training loss
      pred = model(x)
      loss = lossFn(pred, y)
    # zero out the gradients, perform the backpropagation step,
    # and update the weights
      opt.zero_grad()
      loss.backward()
      opt.step()
    # add the loss to the total training loss so far and
    # calculate the number of correct predictions
      totalTrainLoss += loss
      trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()


    with torch.no_grad():
    # set the model in evaluation mode
      model.eval()
    # loop over the validation set
      for (x, y) in valDataLoader:
      # send the input to the device
        (x, y) = (x.to(device), y.to(device))
      # make the predictions and calculate the validation loss
        pred = model(x)
        totalValLoss += lossFn(pred, y)
      # calculate the number of correct predictions
        valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
  # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)
  # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
  # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))
  # we can now evaluate the network on the test set
    print("[INFO] evaluating network...")


 
  


  
 
#plt.savefig(args["plot"])
# serialize the model to disk
#torch.save(model, args["model"])

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  #print ('Returning model... (rollnumber: )')
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = None
  

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  # In addition,
  # Refer to config dict, where learning rate is given, 
  # List of (in_channels, out_channels, kernel_size, stride=1, padding='same')  are specified
  # Example, config = [(1,10,(3,3),1,'same'), (10,3,(5,5),1,'same'), (3,1,(7,7),1,'same')], it can have any number of elements
  # You need to create 2d convoution layers as per specification above in each element
  # You need to add a proper fully connected layer as the last layer
  
  # HINT: You can print sizes of tensors to get an idea of the size of the fc layer required
  # HINT: Flatten function can also be used if required
  return model
  
  
  print ('Returning model... (rollnumber: xx)')
  


# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0


  %matplotlib inline
# loop over our epochs
for e in range(0, EPOCHS):
  # set the model in training mode
  model.train()
  # initialize the total training and validation loss
  totalTrainLoss = 0
  totalValLoss = 0
  # initialize the number of correct predictions in the training
  # and validation step
  trainCorrect = 0
  valCorrect = 0
  # loop over the training set
  for (x, y) in trainDataLoader:
    # send the input to the device
    (x, y) = (x.to(device), y.to(device))
    # perform a forward pass and calculate the training loss
    pred = model(x)
    loss = lossFn(pred, y)
    # zero out the gradients, perform the backpropagation step,
    # and update the weights
    opt.zero_grad()
    loss.backward()
    opt.step()
    # add the loss to the total training loss so far and
    # calculate the number of correct predictions
    totalTrainLoss += loss
    trainCorrect += (pred.argmax(1) == y).type(
      torch.float).sum().item()


  with torch.no_grad():
    # set the model in evaluation mode
    model.eval()
    # loop over the validation set
    for (x, y) in valDataLoader:
      # send the input to the device
      (x, y) = (x.to(device), y.to(device))
      # make the predictions and calculate the validation loss
      pred = model(x)
      totalValLoss += lossFn(pred, y)
      # calculate the number of correct predictions
      valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
  # calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)
  # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
  # print the model training and validation information
  print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
  print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
    avgTrainLoss, trainCorrect))
  print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
    avgValLoss, valCorrect))
  endTime = time.time()
  print("[INFO] total time taken to train the model: {:.2f}s".format(
  endTime - startTime))
  # we can now evaluate the network on the test set
  print("[INFO] evaluating network...")


  # turn off autograd for testing evaluation
  with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # initialize a list to store our predictions
    preds = []
    # loop over the test set
    for (x, y) in testDataLoader:
      # send the input to the device
      x = x.to(device)
      # make the predictions and add them to the list
      pred = model(x)
      preds.extend(pred.argmax(axis=1).cpu().numpy())
  # generate a classification report
    print(classification_report(testData.targets.cpu().numpy(),
    np.array(preds), target_names=testData.classes))


  
  # plot the training loss and accuracy
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(H["train_loss"], label="train_loss")
  plt.plot(H["val_loss"], label="val_loss")
  plt.plot(H["train_acc"], label="train_acc")
  plt.plot(H["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy on Dataset")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend(loc="lower left")
  # set the device we will be using to test the model
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the KMNIST dataset and randomly grab 10 data points
  print("[INFO] loading the KMNIST test dataset...")
  testData = KMNIST(root="data", train=False, download=True,transform=ToTensor())
idxs = np.random.choice(range(0, len(testData)), size=(10,))
testData = Subset(testData, idxs)
# initialize the test data loader
testDataLoader = DataLoader(testData, batch_size=1)
# load the model and set it to evaluation mode
#model = torch.load(args["model"]).to(device)
model.eval()
#plt.savefig(args["plot"])
# serialize the model to disk
#torch.save(model, args["model"])
  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # calculate accuracy, precision, recall and f1score
  
  print ('Returning metrics... (rollnumber: xx)')
  
  return accuracy_val, precision_val, recall_val, f1score_val
