# MLIOT_PROJECT2_INTRUDERALERT
Intruder alert using Passive Infrared Sensor, Camera and a Microphone

One Thing I Learned from audio processing is this:
> "How much MORE THAN normal is considered abnormal?"

So an analogy for AI classifying audio is this:
"What is 1 + 1"?
- A: More than 1 (Confidence = 70%)
- B: Around 2 (Confidence = 20%)
- C - G: Clearly Wrong Answers (Confidence ~2%)

The analogy is that background audio is considered something that is persistent,
but suddenly when there is an anomaly, the background audio still remains (does not disappear), 
it's just that new audio comes in.

So what is the logic for allowing the machine to get the correct answer B?
I thought about fine-tuning (inducing bias on the confidence), but does it really solve the nature of the problem itself?
From the logic, it's good to have a slight scepticism whether 1+1 is truly equals 2, 
That's very human-like, and shouldn't be something that is wrong.

Think of a child solving the math question above, two likely answers are presented. 
How does the child go "I'm confident about the answer A, but this answer B is very disturbing"
