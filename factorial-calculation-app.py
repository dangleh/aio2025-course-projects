import streamlit as st
from factorial import factorial
import os

def main():
    st.title("Factorial Calculation App")
    number = st.number_input("Enter a number:", min_value=0, max_value=900)
    
    if st.button("Calculate"):
        result = factorial(number)
        st.write(f"The factorial of {number} is: {result}")

if __name__ == "__main__":
    main()