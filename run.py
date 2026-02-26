import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
from experiments.exp_recursive_forecasting import Exp_Recursive_Forecast
from experiments.exp_speech_forecasting import Exp_Speech_Forecast
from experiments.exp_mae_pretrain import Exp_MAE_Pretrain
from experiments.exp_mae_finetune import Exp_MAE_Finetune
import random
import numpy as np

if __name__ == "__main__":
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="iTransformer")

    # basic config
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="S_Mamba",
        help="model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer,S_Mamba ]",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="custom", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/electricity/",
        help="root path of the data file",
    )
    parser.add_argument(
        "--data_path", type=str, default="electricity.csv", help="data csv file"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument(
        "--label_len", type=int, default=48, help="start token length"
    )  # no longer needed in inverted Transformers
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )

    # model define
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument(
        "--c_out", type=int, default=7, help="output size"
    )  # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in ecoder",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="whether to predict unseen future data",
    )
    parser.add_argument(
        "--per_variate_scoring",
        action="store_true",
        help="whether to calculate MSE per variate",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument(
        "--loss", type=str, default="MSE", help="loss function, options: [MSE, R2]"
    )
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    )

    # iTransformer
    parser.add_argument(
        "--exp_name",
        type=str,
        required=False,
        default="MTSF",
        help="experiemnt name, options:[MTSF, partial_train, speech]",
    )
    parser.add_argument(
        "--channel_independence",
        type=bool,
        default=False,
        help="whether to use channel_independence mechanism",
    )
    parser.add_argument(
        "--inverse", action="store_true", help="inverse output data", default=False
    )
    parser.add_argument(
        "--class_strategy",
        type=str,
        default="projection",
        help="projection/average/cls_token",
    )
    parser.add_argument(
        "--target_root_path",
        type=str,
        default="./data/electricity/",
        help="root path of the data file",
    )
    parser.add_argument(
        "--target_data_path", type=str, default="electricity.csv", help="data file"
    )
    parser.add_argument(
        "--efficient_training",
        type=bool,
        default=False,
        help="whether to use efficient_training (exp_name should be partial train)",
    )  # See Figure 8 of our paper for the detail
    parser.add_argument(
        "--use_norm", type=int, default=True, help="use norm and denorm"
    )
    parser.add_argument(
        "--partial_start_index",
        type=int,
        default=0,
        help="the start index of variates for partial training, "
        "you can select [partial_start_index, min(enc_in + partial_start_index, N)]",
    )
    parser.add_argument(
        "--d_state", type=int, default=32, help="parameter of Mamba Block"
    )
    parser.add_argument(
        "--temporal_e_layers",
        type=int,
        default=2,
        help="number of temporal Mamba layers (S_Mamba_Speech only)",
    )
    parser.add_argument(
        "--d_conv_temporal",
        type=int,
        default=4,
        help="local conv width for temporal Mamba (S_Mamba_Speech only)",
    )
    parser.add_argument(
        "--d_conv_variate",
        type=int,
        default=4,
        help="local conv width for cross-variate Mamba (S_Mamba_Speech only)",
    )
    parser.add_argument(
        "--expand_temporal",
        type=int,
        default=2,
        help="expansion factor for temporal Mamba (S_Mamba_Speech only)",
    )
    parser.add_argument(
        "--embed_mlp_expand",
        type=int,
        default=2,
        help="expansion factor for MLP embedding hidden dim (S_Mamba_Speech_MLP only)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="max gradient norm for clipping (speech experiment)",
    )
    parser.add_argument(
        "--use_cosine_scheduler",
        action="store_true",
        default=False,
        help="use cosine annealing LR scheduler (speech experiment)",
    )
    parser.add_argument(
        "--recursive_cycles",
        type=int,
        default=10,
        help="number of recursive cycles for recursive forecast experiment",
    )
    parser.add_argument(
        "--checkpoint_model_id",
        type=str,
        default=None,
        help="model id for loading checkpoint in recursive experiment",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=96,
        help="stride for traversing the dataset in recursive experiment",
    )

    # MAE pre-training arguments
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.4,
        help="fraction of frames to mask in MAE pre-training",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=8,
        help="size of contiguous masked blocks in MAE pre-training",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="number of warmup epochs for MAE pre-training scheduler",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="weight decay for AdamW optimizer",
    )

    # Weights & Biases
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="S-D-Mamba",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B team/entity name (default: personal account)",
    )

    # MAE fine-tuning arguments
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default=None,
        help="path to pre-trained MAE encoder checkpoint for fine-tuning",
    )
    parser.add_argument(
        "--finetune_strategy",
        type=str,
        default="full",
        help="fine-tuning strategy: freeze, partial, or full",
    )
    parser.add_argument(
        "--unfreeze_layers",
        type=int,
        default=2,
        help="number of encoder layers to unfreeze in partial fine-tuning",
    )
    parser.add_argument(
        "--lr_encoder",
        type=float,
        default=None,
        help="learning rate for pre-trained encoder (default: 0.1 * learning_rate)",
    )

    args = parser.parse_args()

    # Set lr_encoder default if not specified
    if args.lr_encoder is None:
        args.lr_encoder = args.learning_rate * 0.1
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)
    if args.is_training == 3:
        Exp = Exp_Recursive_Forecast
    elif args.exp_name in (
        "mae_pretrain",
        "transformer_mae_pretrain",
    ):  # MAE self-supervised pre-training
        Exp = Exp_MAE_Pretrain
    elif args.exp_name in (
        "mae_finetune",
        "transformer_mae_finetune",
    ):  # Fine-tune pre-trained MAE for forecasting
        Exp = Exp_MAE_Finetune
    elif args.exp_name == "partial_train":  # See Figure 8 of our paper, for the detail
        Exp = Exp_Long_Term_Forecast_Partial
    elif args.exp_name == "speech":  # Speech time series forecasting
        Exp = Exp_Speech_Forecast
    else:  # MTSF: multivariate time series forecasting
        Exp = Exp_Long_Term_Forecast
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")

    if args.is_training == 1:
        for ii in range(args.itr):
            # setting record of experiments
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy,
                ii,
            )

            # Initialize W&B run
            if args.use_wandb:
                import wandb

                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=setting,
                    config=vars(args),
                    reinit=True,
                )

            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)

            if args.do_predict:
                print(
                    ">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                        setting
                    )
                )
                exp.predict(setting, True)

            if args.use_wandb:
                import wandb

                wandb.finish()

            torch.cuda.empty_cache()
    elif args.is_training == 2:
        print(11111)
        for ii in range(args.itr):
            # setting record of experiments
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy,
                ii,
            )
            exp = Exp(args)
            exp.get_input(setting)
    elif args.is_training == 3:
        ii = 0
        # If checkpoint_model_id is provided, use it for the model loading path (setting)
        # Otherwise use the args.model_id
        model_id_for_loading = (
            args.checkpoint_model_id if args.checkpoint_model_id else args.model_id
        )

        setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
            model_id_for_loading,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy,
            ii,
        )
        exp = Exp(args)
        exp.test_recursive(setting)
        if args.use_wandb:
            import wandb

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"recursive_{setting}",
                config=vars(args),
                reinit=True,
            )
        torch.cuda.empty_cache()
    else:
        ii = 0
        setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy,
            ii,
        )

        exp = Exp(args)  # set experiments
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
