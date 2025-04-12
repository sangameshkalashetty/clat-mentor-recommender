import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Any
import random

class CLATMentorRecommendation:
    def __init__(self):
        self.mentors = []
        self.students = []
        self.feedback_data = []
        self.scaler = StandardScaler()
        self.knn_model = None
        self.feature_weights = {
            'preferred_subjects': 0.3,
            'target_colleges': 0.25,
            'preparation_level': 0.2,
            'learning_style': 0.15,
            'availability': 0.1
        }

    def add_mentor(self, mentor_data: Dict[str, Any]) -> None:
        """Add a new mentor to the system."""
        required_fields = ['id', 'name', 'preferred_subjects', 'target_colleges', 
                         'preparation_level', 'learning_style', 'availability']
        
        if not all(field in mentor_data for field in required_fields):
            raise ValueError("Missing required fields in mentor data")
        
        self.mentors.append(mentor_data)

    def add_student(self, student_data: Dict[str, Any]) -> None:
        """Add a new student to the system."""
        required_fields = ['id', 'name', 'preferred_subjects', 'target_colleges', 
                         'preparation_level', 'learning_style', 'availability']
        
        if not all(field in student_data for field in required_fields):
            raise ValueError("Missing required fields in student data")
        
        self.students.append(student_data)

    def add_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Add feedback from a student-mentor interaction."""
        required_fields = ['student_id', 'mentor_id', 'rating', 'feedback_text']
        
        if not all(field in feedback_data for field in required_fields):
            raise ValueError("Missing required fields in feedback data")
        
        self.feedback_data.append(feedback_data)

    def _vectorize_features(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Convert categorical features to numerical vectors."""
        vectors = []
        
        for item in data:
            vector = []
            
            # Preferred subjects (one-hot encoding)
            subjects = ['english', 'gk', 'legal_aptitude', 'logical_reasoning', 'mathematics']
            subject_vector = [1 if subj in item['preferred_subjects'] else 0 for subj in subjects]
            vector.extend(subject_vector)
            
            # Target colleges (one-hot encoding)
            colleges = ['nls', 'nlud', 'nlu', 'gnlu', 'nliu']
            college_vector = [1 if college in item['target_colleges'] else 0 for college in colleges]
            vector.extend(college_vector)
            
            # Preparation level (ordinal encoding)
            prep_levels = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
            vector.append(prep_levels[item['preparation_level']])
            
            # Learning style (one-hot encoding)
            styles = ['visual', 'auditory', 'reading', 'kinesthetic']
            style_vector = [1 if style in item['learning_style'] else 0 for style in styles]
            vector.extend(style_vector)
            
            # Availability (one-hot encoding)
            times = ['morning', 'afternoon', 'evening', 'weekend']
            time_vector = [1 if time in item['availability'] else 0 for time in times]
            vector.extend(time_vector)
            
            vectors.append(vector)
        
        return np.array(vectors)

    def train_model(self) -> None:
        """Train the KNN model on mentor data."""
        if not self.mentors:
            raise ValueError("No mentor data available for training")
        
        mentor_vectors = self._vectorize_features(self.mentors)
        scaled_vectors = self.scaler.fit_transform(mentor_vectors)
        
        self.knn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
        self.knn_model.fit(scaled_vectors)

    def get_recommendations(self, student_id: str) -> List[Dict[str, Any]]:
        """Get top 3 mentor recommendations for a student."""
        if not self.knn_model:
            raise ValueError("Model not trained. Call train_model() first.")
        
        student = next((s for s in self.students if s['id'] == student_id), None)
        if not student:
            raise ValueError(f"Student with ID {student_id} not found")
        
        student_vector = self._vectorize_features([student])
        scaled_vector = self.scaler.transform(student_vector)
        
        distances, indices = self.knn_model.kneighbors(scaled_vector)
        
        recommendations = []
        for idx, distance in zip(indices[0], distances[0]):
            mentor = self.mentors[idx]
            similarity_score = 1 - distance  # Convert distance to similarity score
            
            # Calculate weighted score based on feature importance
            weighted_score = 0
            for feature, weight in self.feature_weights.items():
                if feature in ['preferred_subjects', 'target_colleges']:
                    # Calculate Jaccard similarity for sets
                    student_set = set(student[feature])
                    mentor_set = set(mentor[feature])
                    intersection = len(student_set.intersection(mentor_set))
                    union = len(student_set.union(mentor_set))
                    feature_similarity = intersection / union if union > 0 else 0
                else:
                    # For other features, use exact match
                    feature_similarity = 1 if student[feature] == mentor[feature] else 0
                
                weighted_score += feature_similarity * weight
            
            recommendations.append({
                'mentor': mentor,
                'similarity_score': similarity_score,
                'weighted_score': weighted_score
            })
        
        # Sort by weighted score and return top 3
        recommendations.sort(key=lambda x: x['weighted_score'], reverse=True)
        return recommendations[:3]

    def update_weights_based_on_feedback(self) -> None:
        """Update feature weights based on feedback data."""
        if not self.feedback_data:
            return
        
        # Calculate average rating for each feature
        feature_ratings = {feature: [] for feature in self.feature_weights.keys()}
        
        for feedback in self.feedback_data:
            student = next((s for s in self.students if s['id'] == feedback['student_id']), None)
            mentor = next((m for m in self.mentors if m['id'] == feedback['mentor_id']), None)
            
            if student and mentor:
                for feature in self.feature_weights.keys():
                    if feature in ['preferred_subjects', 'target_colleges']:
                        student_set = set(student[feature])
                        mentor_set = set(mentor[feature])
                        intersection = len(student_set.intersection(mentor_set))
                        union = len(student_set.union(mentor_set))
                        match_ratio = intersection / union if union > 0 else 0
                    else:
                        match_ratio = 1 if student[feature] == mentor[feature] else 0
                    
                    feature_ratings[feature].append(feedback['rating'] * match_ratio)
        
        # Update weights based on average ratings
        total_weight = 0
        for feature, ratings in feature_ratings.items():
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                self.feature_weights[feature] = avg_rating
                total_weight += avg_rating
        
        # Normalize weights
        for feature in self.feature_weights:
            self.feature_weights[feature] /= total_weight

# Example usage
if __name__ == "__main__":
    # Initialize the recommendation system
    recommender = CLATMentorRecommendation()
    
    # Add sample mentors
    mentors = [
        {
            'id': 'm1',
            'name': 'Aarav Sharma',
            'preferred_subjects': ['legal_aptitude', 'logical_reasoning'],
            'target_colleges': ['nls', 'nlud'],
            'preparation_level': 'advanced',
            'learning_style': ['visual', 'reading'],
            'availability': ['evening', 'weekend']
        },
        {
            'id': 'm2',
            'name': 'Priya Patel',
            'preferred_subjects': ['english', 'gk'],
            'target_colleges': ['nlu', 'gnlu'],
            'preparation_level': 'intermediate',
            'learning_style': ['auditory', 'visual'],
            'availability': ['morning', 'afternoon']
        },
        {
            'id': 'm3',
            'name': 'Rahul Verma',
            'preferred_subjects': ['mathematics', 'logical_reasoning'],
            'target_colleges': ['nliu', 'nls'],
            'preparation_level': 'advanced',
            'learning_style': ['kinesthetic', 'visual'],
            'availability': ['afternoon', 'evening']
        }
    ]
    
    for mentor in mentors:
        recommender.add_mentor(mentor)
    
    # Add sample student
    student = {
        'id': 's1',
        'name': 'Neha Gupta',
        'preferred_subjects': ['legal_aptitude', 'english'],
        'target_colleges': ['nls', 'nlud'],
        'preparation_level': 'intermediate',
        'learning_style': ['visual', 'reading'],
        'availability': ['evening', 'weekend']
    }
    
    recommender.add_student(student)
    
    # Train the model
    recommender.train_model()
    
    # Get recommendations
    recommendations = recommender.get_recommendations('s1')
    
    # Print recommendations
    print("\nTop 3 Mentor Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['mentor']['name']}")
        print(f"   Similarity Score: {rec['similarity_score']:.2f}")
        print(f"   Weighted Score: {rec['weighted_score']:.2f}")
        print(f"   Preferred Subjects: {', '.join(rec['mentor']['preferred_subjects'])}")
        print(f"   Target Colleges: {', '.join(rec['mentor']['target_colleges'])}")
        print(f"   Preparation Level: {rec['mentor']['preparation_level']}")
        print(f"   Learning Style: {', '.join(rec['mentor']['learning_style'])}")
        print(f"   Availability: {', '.join(rec['mentor']['availability'])}")
    
    # Add sample feedback
    feedback = {
        'student_id': 's1',
        'mentor_id': 'm1',
        'rating': 4.5,
        'feedback_text': 'Great mentor, very helpful with legal aptitude'
    }
    recommender.add_feedback(feedback)
    
    # Update weights based on feedback
    recommender.update_weights_based_on_feedback()
    
    print("\nUpdated Feature Weights:")
    for feature, weight in recommender.feature_weights.items():
        print(f"{feature}: {weight:.2f}") 