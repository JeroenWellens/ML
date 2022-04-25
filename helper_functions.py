import matplotlib.pyplot as plt

def plot_loss_curves(history):
    '''
    Returns separate loss curves for training and accuracy metric.
    '''
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"])) # how many epochs did we run for?
    plt.figure(figsize=(8,4))
    plt1=plt.subplot(1,2,1)
    plt2=plt.subplot(1,2,2)
    # Plot loss
    plt1.plot(epochs, loss, label="training_loss")
    plt1.plot(epochs, val_loss, label="val_loss")
    plt1.set_title("loss")
    plt1.legend()

    # Plot accuracy
    plt2.plot(epochs, accuracy, label="training_accuracy")
    plt2.plot(epochs, val_accuracy, label="val_accuracy")
    plt2.set_title("accuracy")
    plt2.legend()