
# Image Segmentation

This project is intended to create a model for semantic segmentation for images containing cell nuclei. The dataset is downloaded from https://www.kaggle.com/competitions/data-science-bowl-2018/overview. This dataset comes as a zip file, and file is splitted into train and test folders. In this project, I am able to train the model with the accuracy of 95%.


## Badges

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)


## Details of Steps

Upload the dataset from Google Drive and unzip the file.

    from google.colab import drive
    drive.mount('/content/gdrive')
    !unzip gdrive/My\ Drive/data-science-bowl-2018.zip

Install tensorflow-examples.

    !pip install git+https://github.com/tensorflow/examples.git

Import packages.

    from tensorflow_examples.models.pix2pix import pix2pix
    from sklearn.model_selection import train_test_split
    from google.colab.patches import cv2_imshow
    from IPython.display import clear_output
    from keras.utils import plot_model
    from tensorflow import keras

    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np
    import cv2,os

- Data Preparation

    Prepare the path.

        root_path = '/content/data-science-bowl-2018-2/train'

    Prepare empty list to hold the data.

        images = []
        masks = []

    Load the images and masks using OpenCV. Resize the image and mask into width and height of (128,128).

        image_dir = os.path.join(root_path,'inputs')
        for image_file in os.listdir(image_dir):
            img = cv2.imread(os.path.join(image_dir,image_file))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(128,128))
            images.append(img)

        masks_dir = os.path.join(root_path,'masks')
        for mask_file in os.listdir(masks_dir):
            mask = cv2.imread(os.path.join(masks_dir,mask_file),cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask,(128,128))
            masks.append(mask)

    Convert the list of np array into a np array.

        images_np = np.array(images)
        masks_np = np.array(masks)

- Data Preprocessing

    Expand the mask dimension.

        masks_np_exp = np.expand_dims(masks_np,axis=-1)
        print(np.unique(masks_np_exp[0]))

    Convert the mask values from [0,255] into [0,1]

        converted_masks = np.round(masks_np_exp / 255.0).astype(np.int64)
        print(np.unique(converted_masks[0]))

    Normalize the images.

        converted_images = images_np/255.0

    Perform train test split on the numpy arrays for the images and masks using scikit-learn.

        SEED = 42
        X_train,X_test,y_train,y_test = train_test_split(converted_images,converted_masks,test_size=0.2,random_state=SEED)

    Convert the numpy arrays into tensor slices using this method: tf.data.Dataset.from_tensor_slices().

        X_train_tensor = tf.data.Dataset.from_tensor_slices(X_train)
        X_test_tensor = tf.data.Dataset.from_tensor_slices(X_test)
        y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
        y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

    Combine the images and masks using the zip method.

        train_dataset = tf.data.Dataset.zip((X_train_tensor,y_train_tensor))
        test_dataset = tf.data.Dataset.zip((X_test_tensor,y_test_tensor))

    Define data augmentation pipeline as a single layer through subclassing.

        class Augment(tf.keras.layers.Layer):
        def __init__(self, seed=42):
            super().__init__()
            # both use the same seed, so they'll make the same random changes.
            self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
            self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

        def call(self, inputs, labels):
            inputs = self.augment_inputs(inputs)
            labels = self.augment_labels(labels)
            return inputs, labels

        #Build the dataset
        BATCH_SIZE = 16
        AUTOTUNE = tf.data.AUTOTUNE
        BUFFER_SIZE = 1000
        TRAIN_SIZE = len(train_dataset)
        STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

        train_batches = (
            train_dataset
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .repeat()
            .map(Augment())
            .prefetch(buffer_size=tf.data.AUTOTUNE))

        test_batches = test_dataset.batch(BATCH_SIZE)

    Visualize some images.

        def display(display_list):
            plt.figure(figsize=(15,15))
            title = ['Input Image','True Mask','Predicted Mask']
            for i in range(len(display_list)):
                plt.subplot(1,len(display_list),i+1)
                plt.title(title[i])
                plt.imshow(keras.utils.array_to_img(display_list[i]))
            plt.show()

        for images,masks in train_batches.take(2):
            sample_image,sample_mask = images[0],masks[0]
            display([sample_image,sample_mask])

- Model Development

    Use a pretrained model as the feature extractor.

        base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)
        base_model.summary()

        #Use these activation layers as the outputs from the feature extractor (some of these outputs will be used to perform concatenation at the upsampling path)
        layer_names = [
            'block_1_expand_relu',      #64x64
            'block_3_expand_relu',      #32x32
            'block_6_expand_relu',      #16x16
            'block_13_expand_relu',     #8x8
            'block_16_project'          #4x4
            ]

        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        #Instantiate the feature extractor
        down_stack = keras.Model(inputs=base_model.input,outputs=base_model_outputs)
        down_stack.trainable = False

        #Define the upsampling path
        up_stack = [
            pix2pix.upsample(512,3),    #4x4 --> 8x8
            pix2pix.upsample(256,3),    #8x8 --> 16x16
            pix2pix.upsample(128,3),    #16x16 --> 32x32
            pix2pix.upsample(64,3)      #32x32 --> 64x64
        ]

        #Use functional API to construct the entire U-net
        def unet(output_channels:int):
            inputs = keras.layers.Input(shape=[128,128,3])
            #Downsample through the model
            skips = down_stack(inputs)
            x = skips[-1]
            skips = reversed(skips[:-1])

            #Build the upsampling path and establish the concatenation
            for up, skip in zip(up_stack,skips):
                x = up(x)
                concat = keras.layers.Concatenate()
                x = concat([x,skip])
            
            #Use a transpose convolution layer to perform the last upsampling, this will become the output layer
            last = keras.layers.Conv2DTranspose(filters=output_channels,kernel_size=3,  strides=2,padding='same') #64x64 --> 128x128
            outputs = last(x)

            model = keras.Model(inputs=inputs,outputs=outputs)

            return model

    Use the function to create the model.

        OUTPUT_CHANNELS = 3
        model = unet(OUTPUT_CHANNELS)
        model.summary()
        keras.utils.plot_model(model)

    Compile the model.

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])

    Create functions to show predictions.

        def create_mask(pred_mask):
            pred_mask = tf.argmax(pred_mask,axis=-1)
            pred_mask = pred_mask[...,tf.newaxis]
            return pred_mask[0]

        def show_predictions(dataset=None,num=1):
            if dataset:
                for image,mask in dataset.take(num):
                    pred_mask = model.predict(image)
                    display([image[0],mask[0],create_mask(pred_mask)])
            else:
                display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))])

        show_predictions()

    Create a callback function to make use of the show_predictions function.

        class DisplayCallback(keras.callbacks.Callback):
            def on_epoch_end(self,epoch,logs=None):
                clear_output(wait=True)
                show_predictions()
                print('\nSample prediction after epoch {}\n'.format(epoch+1))

    Model training.

        EPOCHS = 5
        VAL_SUBSPLITS = 5
        TEST_SIZE = len(test_dataset)
        VALIDATION_STEPS = TEST_SIZE // BATCH_SIZE // VAL_SUBSPLITS

        history = model.fit(train_batches,validation_data=test_batches,validation_steps=VALIDATION_STEPS,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,callbacks=[DisplayCallback()])

    Model training visualization.

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(acc, label='train accuracy')
        plt.plot(val_acc, label='val accuracy')
        plt.title('epoch_accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(loss, label='train loss')
        plt.plot(val_loss, label='val loss')
        plt.title('epoch_loss')
        plt.legend()
        plt.show()

- Model Deployment

    Evaluate the final model.

        test_loss,test_acc = model.evaluate(test_batches)
        print("Loss = ",test_loss)
        print("Accuracy = ",test_acc)

    Visualize the predictions.

        show_predictions(test_batches,3)

    Save the trained model in .h5 file format.

        model.save('model.h5')
## Model Performances

- Trained model accuracy

![accuracy](https://user-images.githubusercontent.com/121777112/212230489-b2fdf32e-33eb-4403-83dc-0492305bab4d.png)

- Model Performance

![model_performances](https://user-images.githubusercontent.com/121777112/212230443-c5dbdd22-ebc8-49ae-9fb0-473cb50ed97e.png)

- Image predictions

![image_prediction](https://user-images.githubusercontent.com/121777112/212230470-185a61dd-e12f-4969-9a2a-7f07aab97d30.png)

- Model architecture

![model_architecture](https://user-images.githubusercontent.com/121777112/212230454-8135b76e-c34d-45d5-8ad0-b1460763c45d.png)



## Acknowledgements

 - [kaggle](https://www.kaggle.com/competitions/data-science-bowl-2018/overview)


