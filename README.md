# MoonWallRating

MoonWallRating is a machine learning project leveraging attention-based models to classify rock climbing routes on the standardized MoonBoard climbing wall. This project demonstrates the potential of machine learning in automating the subjective process of route difficulty rating, a challenge traditionally handled by experienced climbers.

## Project Overview

### Background
The MoonBoard is a globally standardized climbing training wall, offering a large dataset of routes with community-assigned difficulty ratings. This dataset, particularly from the 2017 MoonBoard, serves as the foundation for the study. Traditional methods for route difficulty classification, such as Naive Bayes, CNNs, and LSTMs, have shown limited success, hindered by sparse data representation and class imbalance. 

MoonWallRating adopts transformer-based attention models, inspired by their success in natural language processing, to process climbing route sequences and predict difficulty ratings. The model represents climbing routes as sequences of tokens, capturing key features such as board angle, foothold availability, and hold sequences.

### Results
Key findings and outcomes include:

1. **Performance**: The model achieves an accuracy of 48.8% and a ±1 accuracy of 85.3%, surpassing previous models such as CNNs, GCNs, and Naive Bayes classifiers. It performs slightly better than the GradeNet LSTM model, which achieved 47.5% accuracy and 84.8% ±1 accuracy.

2. **Comparison to Human Performance**: Human experts achieve ±1 accuracy of approximately 87.5% when estimating route difficulty without climbing the routes. MoonWallRating demonstrates that machine learning models can approach and potentially surpass human-level performance in this domain.

3. **Advancements in Data Representation**: The model transitions from traditional grid-based representations of climbing walls to sequence-based encoding, improving its ability to recognize patterns and relationships between holds.

### Practical Implications
The project highlights the potential for machine learning to:

- Assist climbers in selecting routes suited to their skill levels.
- Provide insights to route setters, improving the consistency and objectivity of difficulty ratings.
- Lay the groundwork for future research into automated route generation and personalized climbing training systems.

## Future Directions
MoonWallRating opens avenues for further exploration in:

- Route generation using attention mechanisms to create balanced and challenging climbing problems.
- Expanding datasets to include additional MoonBoard variations and user-generated routes.
- Refining models to handle class imbalances and validate performance across different datasets.

This project underscores the transformative role machine learning can play in modernizing and enriching the climbing experience.
