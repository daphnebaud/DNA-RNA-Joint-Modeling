{
 "cells": [
  {
   "metadata": {},
   "cell_type": "raw",
   "source": "import transformer_model as tf",
   "id": "e806128d44ab1fe5"
  },
  {
   "metadata": {
    "ExecuteTime": {
     "end_time": "2025-06-18T10:57:00.887602Z",
     "start_time": "2025-06-18T10:57:00.625428Z"
    }
   },
   "cell_type": "code",
   "source": [
    "#this is a test data set to see if the model works\n",
    "src_vocab_size = 5000\n",
    "tgt_vocab_size = 5000\n",
    "d_model = 512\n",
    "num_heads = 8\n",
    "num_layers = 6\n",
    "d_ff = 2048\n",
    "max_seq_length = 100\n",
    "dropout = 0.1\n",
    "\n",
    "transformer = tf.Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)\n",
    "\n",
    "# Generate random sample data\n",
    "src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)\n",
    "tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)"
   ],
   "id": "b9ba3a2dad2f90d3",
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'tf' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001B[1;31m---------------------------------------------------------------------------\u001B[0m",
      "\u001B[1;31mNameError\u001B[0m                                 Traceback (most recent call last)",
      "Cell \u001B[1;32mIn[2], line 11\u001B[0m\n\u001B[0;32m      8\u001B[0m max_seq_length \u001B[38;5;241m=\u001B[39m \u001B[38;5;241m100\u001B[39m\n\u001B[0;32m      9\u001B[0m dropout \u001B[38;5;241m=\u001B[39m \u001B[38;5;241m0.1\u001B[39m\n\u001B[1;32m---> 11\u001B[0m transformer \u001B[38;5;241m=\u001B[39m \u001B[43mtf\u001B[49m\u001B[38;5;241m.\u001B[39mTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)\n\u001B[0;32m     13\u001B[0m \u001B[38;5;66;03m# Generate random sample data\u001B[39;00m\n\u001B[0;32m     14\u001B[0m src_data \u001B[38;5;241m=\u001B[39m torch\u001B[38;5;241m.\u001B[39mrandint(\u001B[38;5;241m1\u001B[39m, src_vocab_size, (\u001B[38;5;241m64\u001B[39m, max_seq_length))  \u001B[38;5;66;03m# (batch_size, seq_length)\u001B[39;00m\n",
      "\u001B[1;31mNameError\u001B[0m: name 'tf' is not defined"
     ]
    }
   ],
   "execution_count": 2
  },
  {
   "metadata": {},
   "cell_type": "code",
   "outputs": [],
   "execution_count": null,
   "source": [
    "#training the model\n",
    "criterion = nn.CrossEntropyLoss(ignore_index=0)\n",
    "optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)\n",
    "\n",
    "transformer.train()\n",
    "\n",
    "for epoch in range(100):\n",
    "    optimizer.zero_grad()\n",
    "    output = transformer(src_data, tgt_data[:, :-1])\n",
    "    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "    print(f\"Epoch: {epoch+1}, Loss: {loss.item()}\")"
   ],
   "id": "929d46a136402f12"
  },
  {
   "metadata": {
    "ExecuteTime": {
     "end_time": "2025-06-18T10:56:45.621734Z",
     "start_time": "2025-06-18T10:56:44.403429Z"
    }
   },
   "cell_type": "code",
   "source": [
    "#training model performance evaluation\n",
    "transformer.eval()\n",
    "\n",
    "# Generate random sample validation data\n",
    "val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)\n",
    "val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)\n",
    "\n",
    "with torch.no_grad():\n",
    "\n",
    "    val_output = transformer(val_src_data, val_tgt_data[:, :-1])\n",
    "    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))\n",
    "    print(f\"Validation Loss: {val_loss.item()}\")"
   ],
   "id": "6d610d1c96d41564",
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'transformer' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001B[1;31m---------------------------------------------------------------------------\u001B[0m",
      "\u001B[1;31mNameError\u001B[0m                                 Traceback (most recent call last)",
      "Cell \u001B[1;32mIn[1], line 2\u001B[0m\n\u001B[0;32m      1\u001B[0m \u001B[38;5;66;03m#training model performance evaluation\u001B[39;00m\n\u001B[1;32m----> 2\u001B[0m \u001B[43mtransformer\u001B[49m\u001B[38;5;241m.\u001B[39meval()\n\u001B[0;32m      4\u001B[0m \u001B[38;5;66;03m# Generate random sample validation data\u001B[39;00m\n\u001B[0;32m      5\u001B[0m val_src_data \u001B[38;5;241m=\u001B[39m torch\u001B[38;5;241m.\u001B[39mrandint(\u001B[38;5;241m1\u001B[39m, src_vocab_size, (\u001B[38;5;241m64\u001B[39m, max_seq_length))  \u001B[38;5;66;03m# (batch_size, seq_length)\u001B[39;00m\n",
      "\u001B[1;31mNameError\u001B[0m: name 'transformer' is not defined"
     ]
    }
   ],
   "execution_count": 1
  },
  {
   "metadata": {},
   "cell_type": "code",
   "outputs": [],
   "execution_count": null,
   "source": "",
   "id": "afc85f85eb1a8a1e"
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
