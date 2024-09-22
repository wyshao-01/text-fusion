class args():
    epochs = 20
    batch_size = 4

    datasets_prepared = False

    train_ir = '.\clip picture last/ms_ir'
    train_vi = '.\clip picture last/ms_vi'


    hight = 256
    width = 256
    image_size = 256

    save_model_dir = "models_training"
    save_loss_dir = "loss"

    cuda = 1

    g_lr = 0.0001

    log_interval = 5
    log_iter = 1