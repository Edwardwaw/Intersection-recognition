Project name: Intersection recognition by EfficientNet-d0 or ResNet-50

1.dataset
we orgnize our street data in train_dataset like this:
		root/dog/xxx.png
		root/dog/xxy.png
		root/dog/xxz.png

		root/cat/123.png
		root/cat/nsdf3.png
		root/cat/asd932_.png


2.about our model
ResNet-50 and EfficientNet-d0 are trained here. when training EfficientNet-d0, autoaugmentation policy is used for data agumentation.

3.usage:
how to train an model?
just orgnize your dataset in above format and  set hyperparameters such as lr in train_ResNet.py or train_efficient.py. then run it.

how to test an model?
put your test images in .jpg or .png in the test folder and then run exam.py. results would be written in a .dat file.
