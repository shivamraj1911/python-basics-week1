# Function to calculate grade and message
def calculate_grade(marks):
    if marks >= 90 and marks <= 100:
        return "A", "Excellent work! Keep shining "
    elif marks >= 80:
        return "B", "Very Good! Keep it up "
    elif marks >= 70:
        return "C", "Good effort! You can do even better "
    elif marks >= 60:
        return "D", "You passed! Work harder next time "
    else:
        return "F", "Don't give up! Learn and try again "


# Main program
print(" STUDENT GRADE CALCULATOR ")

# Input student name
student_name = input("Enter student name: ")

# Input marks with validation
while True:
    try:
        marks = int(input("Enter marks (0-100): "))
        if 0 <= marks <= 100:
            break
        else:
            print("Invalid marks! Please enter between 0 and 100.")
    except ValueError:
        print("Please enter numeric marks only.")

# Calculate grade and message
grade, message = calculate_grade(marks)

# Display result
print("\nRESULT FOR", student_name.upper())
print("Marks:", marks, "/100")
print("Grade:", grade)
print("Message:", message)
