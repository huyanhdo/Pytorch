import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import ALL_LETTERS,N_LETTERS
from utils import load_data,letter_to_tensor,line_to_tensor,random_training_example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
	def __init__(self,input_size,hidden_size,output_size):
		super(RNN,self).__init__()
		self.hidden_size = hidden_size
		self.i2h = nn.Linear(input_size + hidden_size,hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size,output_size)
		self.softmax = nn.LogSoftmax(dim = 1)

	def forward(self,x,h):
		combined = torch.cat((x,h),1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)

		output = self.softmax(output)

		return output, hidden

	def init_hidden(self):
		return torch.zeros(1,self.hidden_size)

category_lines,all_category = load_data()

n_categories = len(all_category)

n_hidden = 128
rnn = RNN(N_LETTERS,n_hidden,n_categories).to(device)

def predict_category(output):
	category_idx = torch.argmax(output).item()
	return all_category[category_idx]

crit = nn.NLLLoss()
optim = optim.SGD(rnn.parameters(),lr = 0.001)

def train(line_tensor,category_tensor):
	rnn.train()
	hidden = rnn.init_hidden().to(device)
	line_tensor = line_tensor.to(device)
	category_tensor = category_tensor.to(device)
	for i in range(len(line_tensor)):
		output,hidden = rnn(line_tensor[i],hidden)
	optim.zero_grad()
	l = crit(output,category_tensor)
	l.backward()
	optim.step()
	return output,l.item()

current_loss = 0.0
all_losses = []
plot_steps = 1000
print_steps = 5000
n_iters = 100000
min_loss = 1000000.0
for i in range(n_iters):
	category,line,category_tensor,line_tensor = random_training_example(category_lines,all_category)
	output,loss = train(line_tensor,category_tensor)
	rnn.eval()
	current_loss += loss

	if (i+1) % plot_steps == 0:
	    all_losses.append(current_loss / plot_steps)
	    if (current_loss/len(line_tensor) < min_loss):
	    	min_loss = current_loss/len(line_tensor)
	    	torch.save(rnn.state_dict(),'bestmodel.pth')
	    current_loss = 0
 	
	if (i+1) % print_steps == 0:
		guess = predict_category(output)
		correct = "CORRECT" if guess == category else f"WRONG ({category})"
		print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}") 

plt.figure()
plt.plot(all_losses)
plt.show()

def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line).to(device)
        
        hidden = rnn.init_hidden().to(device)
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        
        guess = predict_category(output)
        print(guess)


predict('hanh')
