import streamlit as st
import os
import sys
sys.path.append('src')
from factorial import factorial
from footer import show_footer

def load_users():
    """read list of users from users.txt"""
    try:
        if os.path.exists("data/users.txt"):
            with open("data/users.txt", "r", encoding="utf-8") as f:
                users = [line.strip() for line in f.readlines() if line.strip()]
            return users
        else:
            st.error("Users.txt is not found!")
            return []
    except Exception as e:
        st.error(f"Error reading users.txt: {e}")
        return []
    
def login_page():
    """Login page"""
    st.title("Login")

    # Input username
    username = st.text_input("Enter username:")
    
    if st.button("Login"):
        if username:
            users = load_users()
            if username in users:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                # If user not found, show greeting
                st.session_state.show_greeting = True
                st.session_state.username = username
                st.rerun()
        else:
            st.warning("Please enter a username!")
    
    show_footer()

def factorial_calculator():
    """Factorial calculator page"""
    st.title("Factorial Calculator")
    
    # Hiển thị thông tin user đã đăng nhập
    st.write(f"Hello, {st.session_state.username}!")
    
    # Nút đăng xuất
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
    
    st.divider()
    
    number = st.number_input("Enter a number:", 
                              min_value=0, 
                              max_value=900)

    if st.button("Calculate"):
        result = factorial(number)
        st.write(f"The factorial of {number} is: {result}")

    show_footer()

def greeting_page():
    """Greeting page for new users"""
    st.title("Welcome!")
    st.write(f"Hello {st.session_state.username}!")
    st.write("You do not have access to the factorial calculation app.")

    if st.button("Back to Login"):
        st.session_state.show_greeting = False
        st.session_state.username = ""
        st.rerun()

    show_footer()

def main():
    # ARCHIVED 
    # st.title("Factorial Calculation App")
    # number = st.number_input("Enter a number:", min_value=0, max_value=900)
    
    # if st.button("Calculate"):
    #     result = factorial(number)
    #     st.write(f"The factorial of {number} is: {result}")

    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'show_greeting' not in st.session_state:
        st.session_state.show_greeting = False
    
    # Check login status and show appropriate page
    if st.session_state.logged_in:
        factorial_calculator()
    elif st.session_state.show_greeting:
        greeting_page()
    else:
        login_page()


if __name__ == "__main__":
    main()