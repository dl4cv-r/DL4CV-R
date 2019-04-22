Please add information about the project here.

**Idea for learned data preprocessing.** 

1. Multiply the data by some large number (e.g. 2^20).  
Make sure that the rough range of the data is between 0~1.

2. Add an instance normalization layer in the beginning of the model.   
This allows the model to decide how much to shrink, expand, shift the data values.

3. If a boolean needs to be concatenated to the input, add it after the instance norm.  
This way, it is not shifted. 
(Though this does not seem to matter now that I think about it)  
On second thoughts, just concatenate before putting it into the model.  
It won't matter whether the boolean is shifted or not.

4. The labels should be multiplied by the same constant as the data.  
The results should be divided by the same value for reconstruction.


**Discussion**
 

1. This might be better since it reduces the slice-wise dependency in 
dividing each slice by its standard deviation and subtracting by its bias.  
For example, small insignificant slices in front have small values and high standard deviations, 
especially when down-sampled.  
Putting them in the same range as data preprocessing seems to be wrong. 
This might help reduce their influence.  
This way, the whole block/dataset will be preprocessed similarly.

2. The data preprocessing step becomes simpler. Just one constant is needed.

3. Using this method the multiplied constant need not be fine-tuned much since the 
Instance norm layer will correct for moderate scaling. 

4. I am still uncertain about subtracting the mean of each slice before putting it into the model, 
though it seems to produce good results.   
Although since the labels are also subtracted, 
the fact that each subtracted value is different might not matter much.


**Second Thoughts**  
On second thoughts, if the data is normalized by instance normalization anyway.  
Therefore, it is better to normalize before going into the model since this way,  
there will be no need for the model to guess the unknowable bias of the initial input,  
which has been lost in the normalization process. 
