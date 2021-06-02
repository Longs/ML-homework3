import numpy as np
def perceptron_origin(data,labels,T):
    answer = []
    for _ in range(len(data)):
        answer.append(0)
    changed = True
    iteration = 0
    data_t = data.T
    #print(f"data: {data}")
    #print(f"data_T: {data_t}")

    #adding printed count of mistakes for homework 3
    mistakes = 0

    while changed !=False:
        changed = False
        if iteration == T: 
            print(f"mistakes: {mistakes}")
            return answer # - there is no right answer but this will do...
        for i in range (len(data[0])):
            #print(np.sign(np.dot(answer,data[i])))
            #print(np.sign(labels[0][i]))
            if np.sign(np.dot(answer,data_t[i])) != np.sign(labels[0][i]):
                answer = answer + labels[0][i]*data_t[i]
                changed = True
                mistakes +=1
        iteration += 1

    print(f"mistakes: {mistakes}")
    return answer

h3_data=np.array([[200,800,200,800],[0.2,0.2,0.8,0.8],[1,1,1,1]])
h3_labels = np.array([[-1,-1,1,1]])

print(f"***************** h3 original:*************** ")
print(f"h3_data = {h3_data}")
print(f"answer: {perceptron_origin(h3_data,h3_labels,-1)}")

for _ in range (len(h3_data[0])):
    h3_data[0,_] = h3_data[0,_] * 0.001

print(f"***************** h3 MODIFIED:*************** ")
print(f"h3_data = {h3_data}")
print(f"answer: {perceptron_origin(h3_data,h3_labels,-1)}")


def add_offset_row(data):
    """
    add row of 1s to data to ensure value of theta0 offset is output
    """
    #new_row = []
    new_row = []
    #TODO: is there a more efficient way?
    #for _ in range(len(data[0])):
    #    new_row.append(1)

    #new_row = np.array(new_row, ndmin=2)
    
    new_row.append(np.ones(len(data[0])))
    new_row = np.array(new_row)

    return np.concatenate((data,new_row),axis=0)

def perceptron(data,labels, params={}, hook=None):

    T = params.get('T', 100)
    #add 1 extra dimension with 1 to record the offset
    #data is stored as dxn so need to add a new row
    new_data = add_offset_row(data)

    """ #following replaced by add_offset_row
    new_row = []
    #TODO: is there a more efficient way?
    for _ in range(len(data[0])):
        new_row.append(1)

    new_row = np.array(new_row, ndmin=2)
    #print(f"data: {data}")
    #print(f"new_row {new_row}")
    new_data=np.concatenate((data,new_row),axis=0)
    """

    
    #print(f"new_data: {new_data}")
    #print(f"labels: {labels}")
    #print(f"")

    origin_answer = perceptron_origin(new_data,labels,T)
    #print(f"origin_answer {origin_answer}")
    if origin_answer is None:
        return np.array([[1,1]]), np.array([[1]])
    else: 
        return origin_answer[:-1].reshape((len(origin_answer[:-1]),1)), np.array(origin_answer[-1],ndmin=2)

#Visualization of perceptron, comment in the next three lines to see your perceptron code in action:
'''
for datafn in (super_simple_separable_through_origin,super_simple_separable):
   data, labels = datafn()
   test_linear_classifier(datafn,perceptron,draw=True)
'''

print(f"&&&&&&&&&&&&&& homework 3 q 2 &&&&&&&&&&&&&&&&")
h3_data=np.array([[2,3,4,5]])
h3_labels = np.array([[1,1,-1,-1]])

print(f"answer: {perceptron(h3_data,h3_labels)}")

def one_hot(x, k):
    out = np.zeros((k,1))
    out[x-1][0]=1
    return out

print(f"&&&&&&&&&&&&&& homework 3 q 2E &&&&&&&&&&&&&&&&")

new_h3_data=[]
#for _ in h3_data[0]:
#    new_h3_data.append (one_hot(_,6))

# doinmg manually for now since its more complex than^^ this ^^
new_h3_data=np.array([[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0],[1,1,1,1]])

print(f"data: {new_h3_data}")
print(f"answer: {perceptron_origin(new_h3_data,h3_labels,100)}")

print(f"&&&&&&&&&&&&&& homework 3 q 2F &&&&&&&&&&&&&&&&")

new_h3_data=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[1,1,1,1,1,1]])
h3_labels = np.array([[1,1,-1,-1,1,1]])
print(f"data: {new_h3_data}")
print(f"answer: {perceptron_origin(new_h3_data,h3_labels,100)}")