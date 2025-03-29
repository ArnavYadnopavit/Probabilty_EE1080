import numpy as np
import argparse
import matplotlib.pyplot as plt
np.random.seed(41)
def mode0Angle(N):
    Uniform_Angles=np.random.uniform(0,np.pi,N)
    Chord_length=2*np.sin(Uniform_Angles)
    bernoulli=(Chord_length>np.sqrt(3)).astype(int)
    #1 if in the range else 0
    mean=np.mean(bernoulli)
    print("Fraction of the chord length samples that are greater than $\sqrt{3}$= ",f"{mean:.3f}")
    plt.hist(Chord_length,bins=int(np.sqrt(N)))
    plt.xlabel("Chord Length Samples")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Chord Length Samples")
    plt.show()

def mode1Distance(N):
    Uniform_Distance=np.random.uniform(0,1,N)
    Chord_length=2*(np.sqrt(1-Uniform_Distance**2))
    bernoulli=(Chord_length>np.sqrt(3)).astype(int)
    #1 if in the range else 0
    mean=np.mean(bernoulli)
    print("Fraction of the chord length samples that are greater than $\sqrt{3}$= ",f"{mean:.3f}")
    plt.hist(Chord_length,bins=int(np.sqrt(N)))   
    plt.xlabel("Chord Length Samples")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Chord Length Samples")    
    plt.show()

def mode2Centre(N):
    #X=np.random.uniform(-1,1,N)
    #Y=np.random.uniform(-1,1,N)
    #Cancelled the above approach due to the coordinates forming square instead of circle
    theta = np.random.uniform(0, 2*np.pi, N)
    r= np.sqrt(np.random.uniform(0, 1, N))  
    X= r * np.cos(theta)
    Y= r * np.sin(theta)
    R=np.sqrt(Y**2+X**2)
    Chord_length_Z=2*(np.sqrt(1-R**2))     
    bernoulliR=(Chord_length_Z>np.sqrt(3)).astype(int)
    #1 if in the range else 0
    mean=np.mean(bernoulliR)
    print("Fraction of the chord length samples that are greater than $\sqrt{3}$= ",f"{mean:.3f}")
    plt.hist(Chord_length_Z,bins=int(np.sqrt(N)))       
    plt.xlabel("Chord Length Samples")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Chord Length Samples")    
    plt.show()



parser = argparse.ArgumentParser()
parser.add_argument("mode",type=int,help="mode")
parser.add_argument("N",type=int,help="N")
args = parser.parse_args()

if args.mode==0:
    mode0Angle(args.N) #Refers to part 4.1
elif args.mode==1:
    mode1Distance(args.N) #Refers to part 4.2
elif args.mode==2:
    mode2Centre(args.N) #Refers to part 4.3
else:
    print("Onle mode 0,1 and 2 exist")




