def add_train_args(parser):
    #train_args = parser.add_argument_group("Training Options",
    #                                      "Configurations options for RSR-GAN training")
    parser.add_argument('--train-manifest', metavar='DIR',
                        help='path to train manifest csv', default='data/train_manifest.csv')
    parser.add_argument('--val-manifest', metavar='DIR',
                        help='path to validation manifest csv', default='data/val_manifest.csv')
    parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
    parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
    parser.add_argument('--labels-path', default='labels_dict.json', help='Contains all characters for transcription')
    parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
    parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
    parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')

    parser.add_argument('--enc-hid-dim', default=256, type=int, help='Encoder hidden dimension')
    parser.add_argument('--dec-hid-dim', default=256, type=int, help='Decoder hidden dimension')
    parser.add_argument('--dec-emb-dim', default=256, type=int, help='Decoder embedding dimension')
    parser.add_argument('--dropout-rate', default=0.2, type=float, help='Dropout rate')

    parser.add_argument('--epochs', default=500, type=int, help='Number of training epochs')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
    parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
    parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
    parser.add_argument('--log-dir', default='visualize/rsrgan_final', help='Location of tensorboard log')
    parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
    parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
    parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
    parser.add_argument('--model-path', default='models/rsrgan_final.pth',
                        help='Location to save best validation model')
    parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
    parser.add_argument('--finetune', dest='finetune', action='store_true',
                        help='Finetune the model from checkpoint "continue_from"')

    parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                        help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
    
    return parser