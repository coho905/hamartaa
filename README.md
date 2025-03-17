# hamartaa
adversarially transform images to interfere with image recognition algorithms

hamartaa is the anglicization of the Finnish word for blur.

## Example:

### Results after hamartaa on African Elephant:

Final prediction: Class 385 with 1.0000 confidence <br>
Attack successful <br>
True class: African elephant, Loxodonta africana<br>
Predicted class: Indian elephant, Elephas maximus

![Elephant](photos/elephant2.jpg)
->
![Elephant](photos/elephant_adverse.jpg)

### Results after hamartaa on German Shephard
Final prediction: Class 270 with 1.0000 confidence <br>
Attack successful <br>
True class: German shepherd, German shepherd dog, German police dog, alsatian <br>
Predicted class: white wolf, Arctic wolf, Canis lupus tundrarum <br>

![German Shep](photos/german_shep.jpg)
->
![German Shep](photos/adversarial_german_shep.jpg)
