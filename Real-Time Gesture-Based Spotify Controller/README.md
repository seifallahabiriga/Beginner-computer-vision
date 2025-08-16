# Real-Time Gesture-Based Spotify Controller

## Overview  
This project enables **real-time hand gesture recognition** to control Spotify playback or local media commands without needing to pause a game or leave the active screen.  

The idea originated while playing competitive racing games: pausing to adjust music volume, skip tracks, or pause playback broke focus and immersion. The solution was to offload these controls to **computer vision + hand gestures**, making the experience seamless.  

There are **two versions** of the controller provided:
1. **`gesture_controller_using_api.py`** – Uses Spotify Web API for commands (requires authentication).  
2. **`gesture_controller_local.py`** – Uses local OS keyboard shortcuts to control media (works without Spotify API, suitable for any media player supporting system shortcuts).  

---

## Features and Commands  
Gestures are recognized in real time via OpenCV and MediaPipe, then mapped to playback actions:

- **High Five** → Pause playback  
- **Thumbs Up** → Play playback  
- **Left Head Gesture** → Previous track  
- **Right Head Gesture** → Next track  
- **Left Hand distance control** → Adjust system volume dynamically  

---

## Metrics and Logic Used
1. **Distance Metric**  
   - The distance between the hand and camera (normalized to a max threshold) is used to set **system volume level**.  

2. **Angle Metric**  
   - Head tilt can be measured for additional gesture differentiation.  
   - Useful if expanding to more commands (e.g., brightness or window switching).  

3. **Gesture Classification**  
   - MediaPipe landmarks are analyzed to classify gestures (`High Five`, `Thumbs Up`, `Left`, `Right`).  
   - The classifier is lightweight and real-time, enabling smooth gameplay experience.  
