from model import BPRModel
from prepare import collate_fn, tokenizer, train_dataset, valid_dataset
from transformers import Trainer, TrainingArguments

# パラメータ
training_args = TrainingArguments(
    output_dir="outputs_bpr",
    per_device_train_batch_size=32,
    per_gpu_eval_batch_size=32,
    learning_rate=1e-5,
    max_grad_norm=2.0,
    num_train_epochs=1,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,
    remove_unused_columns=False,
)

# モデルの初期化
model = BPRModel("cl-tohoku/bert-base-japanese-v3")

# Trainerの初期化
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# 訓練
trainer.train()

# モデルの保存
question_encoder_path = "outputs_bpr/question_encoder"
model.question_encoder.save_pretrained(question_encoder_path)
tokenizer.save_pretrained(question_encoder_path)

passage_encoder_path = "outputs_bpr/passage_encoder"
model.passage_encoder.save_pretrained(passage_encoder_path)
tokenizer.save_pretrained(passage_encoder_path)
