model = dict(
    type='PAN_PP',
    backbone=dict(
        type='resnet18',
        pretrained=True
    ),
    neck=dict(
        type='FPEM_v2',
        in_channels=(64, 128, 256, 512),
        out_channels=128
    ),
    detection_head=dict(
        type='PAN_PP_DetHead',
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_emb=dict(
            type='EmbLoss_v2',
            feature_dim=4,
            loss_weight=0.25
        ),
        use_coordconv=False,
    ),
    recognition_head=dict(
        type='PAN_PP_RecHead_CTC',
        input_dim=512,
        hidden_dim=128,
        feature_size=(8, 32)    # 截取的识别输入之前的特征的大小
    ),
    description_head=dict(
        type='PAN_PP_DescHead',
        vector_dim=128,
        img_size=736,
        loss_weight=1.0
    ),
    use_mulloss=False
)

data = dict(
    batch_size=48,
    num_workers=16,
    train=dict(
        type='PAN_PP_COCOText',
        split='train',
        is_transform=True,
        img_size=model['description_head']['img_size'],    # 训练的输入尺寸(736, 736)
        short_size=model['description_head']['img_size'],
        kernel_scale=0.5,
        read_type='cv2',
        with_rec=True,
        with_desc=True,
        direction_aug=True,
        direction_ratio=0.3,
        with_kwaitrain=False,     # 是否使用kuwaitrain训练数据，因为存在单字标注噪音
    ),
    test=dict(
        type='PAN_PP_VAL',
        split='test',
        short_size=736,
        read_type='cv2',
        with_rec=True,
        align_mode='short'
    )
)
train_cfg = dict(
    lr=1e-4,
    schedule='polylr',
    epoch=100,
    optimizer='Adam',
    use_ex=False,
    freeze_backbone=False,
    freeze_neck=False,
    freeze_det=False,
    freeze_rec=False,
    freeze_desc=True,
    load_desc_weights=False,
    isvalidate=False
)
test_cfg = dict(
    min_score=0.88,
    min_rec_score=0.15,
    min_area=260,
    min_kernel_area=2.6,
    is_half = False,
    scale=4,
    bbox_type='rect',
    result_path='outputs/r18_ctc_alldata'
)
