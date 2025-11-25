class TinyStoriesDataset:

    def __init__(self, dataset, tokenizer, max_length=512, max_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples if max_samples is not None else len(dataset)
    
    def __len__(self):
        return self.max_samples
    
    def __getitem__(self, idx):
        # Get raw text
        text = self.dataset[idx]['text']
        
        # Tokenize on the fly
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'  # Return tensors for direct use
        )
        
        # Create labels (same as input_ids for causal language modeling)
        labels = encoded['input_ids'].clone()
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),  # Remove batch dimension
            'labels': labels.squeeze(0)  # Remove batch dimension
        }
