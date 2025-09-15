# KCRH_App.py (Updated with sidebar persistence + session_state driver handling)
# Kisumu County Referral Hospital - Streamlit App
# Combines referral system, ambulance tracking, communications, handover forms,
# mapping, analytics and offline sync (demo / simulation version).

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
from typing import List, Dict

# Mapping & geolocation
import folium
from geopy.distance import geodesic
import streamlit.components.v1 as components

# Visualization
import matplotlib.pyplot as plt

# Faker for simulated content
from faker import Faker
faker = Faker()

# -----------------------------
# Core data classes
# -----------------------------
class Hospital:
    def __init__(self, name, location, capacity, hospital_type="general"):
        self.name = name
        self.location = location  # (lat, lon)
        self.capacity = capacity
        self.available_beds = capacity
        self.type = hospital_type
        self.referrals_received = []

    def admit_patient(self, patient):
        if self.available_beds > 0:
            self.available_beds -= 1
            self.referrals_received.append(patient)
            patient.status = "admitted"
            return True
        return False

    def discharge_patient(self, patient):
        if patient in self.referrals_received:
            self.referrals_received.remove(patient)
            self.available_beds += 1
            patient.status = "discharged"
            return True
        return False


class Patient:
    def __init__(self, name, condition, severity, vital_signs=None):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.vital_signs = vital_signs or {}
        self.status = "waiting"
        self.transfer_completion_time = None


class Ambulance:
    def __init__(self, amb_id, location):
        self.id = amb_id
        self.location = location
        self.status = "available"  # available, dispatched, en_route, arrived
        self.current_patient = None
        self.destination = None
        self.route = []
        self.eta = None

    def dispatch(self, patient, destination):
        self.status = "dispatched"
        self.current_patient = patient
        self.destination = destination

    def complete_transfer(self):
        self.status = "available"
        if self.current_patient:
            self.current_patient.transfer_completion_time = datetime.datetime.now()
        self.current_patient = None
        self.destination = None


# -----------------------------
# Specialized systems
# -----------------------------
class ReferralSystem:
    def __init__(self):
        self.hospitals: List[Hospital] = []
        self.ambulances: List[Ambulance] = []
        self.referral_requests: List[Dict] = []
        self.referral_history = pd.DataFrame(columns=[
            "Patient", "From Hospital", "To Hospital", "Ambulance",
            "Status", "Request Time", "Completion Time"
        ])

    def add_hospital(self, hospital: Hospital):
        self.hospitals.append(hospital)

    def add_ambulance(self, ambulance: Ambulance):
        self.ambulances.append(ambulance)

    def find_available_ambulance(self):
        return next((amb for amb in self.ambulances if amb.status == "available"), None)

    def create_referral(self, patient: Patient, from_hospital: Hospital, to_hospital: Hospital, ambulance: Ambulance=None):
        amb = ambulance or self.find_available_ambulance()
        if not amb:
            return None

        referral = {
            "id": len(self.referral_requests) + 1,
            "patient": patient,
            "from_hospital": from_hospital,
            "to_hospital": to_hospital,
            "ambulance": amb,
            "timestamp": datetime.datetime.now(),
            "status": "in_transit"
        }

        amb.dispatch(patient, to_hospital)
        self.referral_requests.append(referral)
        return referral

    def complete_referral(self, referral_id: int):
        referral = next((r for r in self.referral_requests if r["id"] == referral_id), None)
        if not referral:
            return None

        referral["ambulance"].complete_transfer()
        referral["status"] = "completed"
        completion_time = datetime.datetime.now()

        new_entry = {
            "Patient": referral["patient"].name,
            "From Hospital": referral["from_hospital"].name,
            "To Hospital": referral["to_hospital"].name,
            "Ambulance": referral["ambulance"].id,
            "Status": referral["status"],
            "Request Time": referral["timestamp"],
            "Completion Time": completion_time
        }
        self.referral_history = pd.concat([self.referral_history, pd.DataFrame([new_entry])], ignore_index=True)
        return referral


class AmbulanceTracker:
    def __init__(self, system: ReferralSystem):
        self.system = system

    def calculate_distance(self, loc1, loc2):
        return geodesic(loc1, loc2).km

    def simulate_movement(self, ambulance: Ambulance, dest_loc, speed_kmh=60):
        distance_km = self.calculate_distance(ambulance.location, dest_loc)
        minutes = (distance_km / speed_kmh) * 60
        ambulance.eta = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
        ambulance.route = self.generate_route(ambulance.location, dest_loc)
        ambulance.status = "en_route"
        return {"distance_km": distance_km, "eta": ambulance.eta}

    def generate_route(self, start, end, num_points=6):
        lat_diff = (end[0] - start[0]) / (num_points + 1)
        lon_diff = (end[1] - start[1]) / (num_points + 1)
        pts = []
        for i in range(1, num_points + 1):
            pts.append((start[0] + lat_diff * i + random.uniform(-0.0005, 0.0005),
                        start[1] + lon_diff * i + random.uniform(-0.0005, 0.0005)))
        return pts


class CommunicationSystem:
    def __init__(self):
        self.messages = []

    def send_message(self, sender, recipient, message_type, content, urgent=False):
        msg = {
            "id": len(self.messages) + 1,
            "sender": sender,
            "recipient": recipient,
            "type": message_type,
            "content": content,
            "urgent": urgent,
            "timestamp": datetime.datetime.now(),
            "read": False
        }
        self.messages.append(msg)
        return msg

    def get_messages_for(self, recipient):
        return [m for m in self.messages if m["recipient"] == recipient]


class DigitalHandoverSystem:
    def __init__(self):
        self.forms = []

    def create_handover(self, referral):
        patient = referral["patient"]
        form = {
            "form_id": f"FORM_{len(self.forms)+1:06d}",
            "patient": patient.name,
            "condition": patient.condition,
            "sending": referral["from_hospital"].name,
            "receiving": referral["to_hospital"].name,
            "vitals": patient.vital_signs,
            "created": datetime.datetime.now(),
            "status": "draft"
        }
        self.forms.append(form)
        return form


class OfflineManager:
    def __init__(self):
        self.offline_queue = []
        self.sync_history = []
        self.is_online = True

    def go_offline(self):
        self.is_online = False

    def go_online(self):
        self.is_online = True

    def queue(self, action_type, data):
        item = {
            "id": len(self.offline_queue) + 1,
            "action_type": action_type,
            "data": data,
            "timestamp": datetime.datetime.now(),
            "status": "queued"
        }
        self.offline_queue.append(item)
        return item

    def sync(self):
        if not self.is_online:
            return False
        while self.offline_queue:
            item = self.offline_queue.pop(0)
            item["status"] = "synced"
            item["synced_at"] = datetime.datetime.now()
            self.sync_history.append(item)
        return True


# -----------------------------
# Initialize system data (Kisumu project specifics)
# -----------------------------
HOSP_COORDS = {
    "Kisumu County Referral Hospital": (-0.10129, 34.75598),
    "Jaramogi Oginga Odinga Teaching & Referral Hospital": (-0.08864, 34.7714),
    "Ahero Sub-County Hospital": (-0.1711, 34.9175),
    "Kombewa Sub-County Hospital": (-0.1182, 34.5958),
    "Nyabondo Trauma Centre": (-0.1859, 34.8102),
    "Lumumba Sub-County Hospital": (-0.0981, 34.7506),
    "Pap Onditi Sub-County Hospital": (-0.1964, 34.9033),
    "Chulaimbo Sub-District Hospital": (-0.0743, 34.6456)
}

kisumu = Hospital("Kisumu County Referral Hospital", HOSP_COORDS["Kisumu County Referral Hospital"], 200, "referral")
jaramogi = Hospital("Jaramogi Oginga Odinga Teaching & Referral Hospital", HOSP_COORDS["Jaramogi Oginga Odinga Teaching & Referral Hospital"], 400, "teaching")
ahero = Hospital("Ahero Sub-County Hospital", HOSP_COORDS["Ahero Sub-County Hospital"], 120, "sub-county")
kombewa = Hospital("Kombewa Sub-County Hospital", HOSP_COORDS["Kombewa Sub-County Hospital"], 100, "sub-county")
nyabondo = Hospital("Nyabondo Trauma Centre", HOSP_COORDS["Nyabondo Trauma Centre"], 80, "trauma")
lumumba = Hospital("Lumumba Sub-County Hospital", HOSP_COORDS["Lumumba Sub-County Hospital"], 90, "sub-county")
pap_onditi = Hospital("Pap Onditi Sub-County Hospital", HOSP_COORDS["Pap Onditi Sub-County Hospital"], 70, "sub-county")
chulaimbo = Hospital("Chulaimbo Sub-District Hospital", HOSP_COORDS["Chulaimbo Sub-District Hospital"], 85, "sub-district")

if 'ref_sys' not in st.session_state:
    st.session_state.ref_sys = ReferralSystem()
ref_sys = st.session_state.ref_sys

if not ref_sys.hospitals:
    ref_sys.add_hospital(kisumu)
    ref_sys.add_hospital(jaramogi)
    ref_sys.add_hospital(ahero)
    ref_sys.add_hospital(kombewa)
    ref_sys.add_hospital(nyabondo)
    ref_sys.add_hospital(lumumba)
    ref_sys.add_hospital(pap_onditi)
    ref_sys.add_hospital(chulaimbo)

if 'ambulances_created' not in st.session_state:
    ambulances = [Ambulance(f"KSM-AMB-{i+1:03d}", kisumu.location) for i in range(3)]
    for a in ambulances:
        ref_sys.add_ambulance(a)
    st.session_state.ambulances_created = True

if 'patients' not in st.session_state:
    st.session_state.patients = [
        Patient("John Otieno", "trauma", 4, {"bp": "110/70", "hr": 110}),
        Patient("Mary Achieng'", "maternity", 3, {"bp": "120/80", "hr": 90}),
        Patient("Beatrice Ayako", "cardiac", 5, {"bp": "90/60", "hr": 140})
    ]
patients = st.session_state.patients

# Helper: convert referral history to dataframe for display
def referrals_to_df(system: ReferralSystem):
    if system.referral_history.empty:
        return pd.DataFrame(columns=["Patient", "From Hospital", "To Hospital", "Ambulance", "Status", "Request Time", "Completion Time"])
    return system.referral_history.copy()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="KCRH Referral System", layout="wide")
st.title("🏥 Kisumu County Referral Hospital - Referral System (Demo)")

# Sidebar with persistence
if "sidebar_choice" not in st.session_state:
    st.session_state.sidebar_choice = "Dashboard"

sidebar_options = ["Dashboard", "Create Referral", "Map & Routes", "Ambulances", "Communications", "Handover Forms", "Offline"]

sidebar_choice = st.sidebar.selectbox(
    "Navigation",
    sidebar_options,
    index=sidebar_options.index(st.session_state.sidebar_choice),
    key="sidebar_choice"
)

# -----------------------------
# Dashboard
# -----------------------------
if sidebar_choice == "Dashboard":
    st.header("📊 Dashboard")
    st.subheader("Hospitals")
    cols = st.columns(3)
    for i, h in enumerate(ref_sys.hospitals):
        with cols[i % 3]:
            st.write(f"**{h.name}**")
            st.write(f"Type: {h.type}")
            st.write(f"Capacity: {h.capacity}")
            st.progress((h.capacity - h.available_beds) / h.capacity if h.capacity > 0 else 0)

    st.subheader("Referral History")
    df_hist = referrals_to_df(ref_sys)
    st.dataframe(df_hist)


# -----------------------------
# Create Referral
# -----------------------------
elif sidebar_choice == "Create Referral":
    st.header("➕ Create Referral")
    sel = st.radio("Choose patient", [p.name for p in patients] + ["Add new patient"])
    if sel == "Add new patient":
        gen_name = st.text_input("Patient name")
        gen_condition = st.selectbox("Condition", ["trauma", "cardiac", "maternity", "stroke", "pediatric"])
        gen_severity = st.slider("Severity", 1, 5, 3)
        if st.button("Create patient"):
            if gen_name.strip() == "":
                st.error("Please enter the patient's name")
            else:
                newp = Patient(gen_name, gen_condition, gen_severity, {"bp": "--/--"})
                patients.append(newp)
                st.success(f"Patient {gen_name} created")
    else:
        patient_obj = next(p for p in patients if p.name == sel)

    # Step 1: Referral INTO KCRH
    from_hosp_name = st.selectbox(
        "Select referring hospital",
        [
            "Ahero Sub-County Hospital",
            "Kombewa Sub-County Hospital",
            "Nyabondo Trauma Centre",
            "Lumumba Sub-County Hospital",
            "Pap Onditi Sub-County Hospital",
            "Chulaimbo Sub-District Hospital",
        ]
    )
    from_hosp = next(h for h in ref_sys.hospitals if h.name == from_hosp_name)

    # Step 2: KCRH is central hub
    st.markdown("➡️ **Referral first goes to Kisumu County Referral Hospital (KCRH)**")

    # Step 3: Option to escalate to JOOTRH
    escalate = st.checkbox("Escalate referral to Jaramogi Oginga Odinga Teaching & Referral Hospital (JOOTRH)")

    if st.button("Initiate Referral"):
        try:
            if escalate:
                # Two-step referral: from_hosp -> KCRH -> JOOTRH
                ref1 = ref_sys.create_referral(patient_obj, from_hosp, kisumu)
                ref2 = ref_sys.create_referral(patient_obj, kisumu, jaramogi)
                if ref1 and ref2:
                    st.success(f"Referral escalated from {from_hosp.name} ➝ KCRH ➝ JOOTRH using {ref2['ambulance'].id}")
                else:
                    st.error("No available ambulances for this escalation.")
            else:
                # Normal referral into KCRH
                referral = ref_sys.create_referral(patient_obj, from_hosp, kisumu)
                if referral:
                    st.success(f"Referral created from {from_hosp.name} ➝ KCRH with ambulance {referral['ambulance'].id}")
                else:
                    st.error("No available ambulances at the moment.")
        except Exception as e:
            st.error(str(e))

# -----------------------------
# Map & Routes
# -----------------------------
elif sidebar_choice == "Map & Routes":
    st.subheader("🚑 Ambulance Tracking & Routes (Simulation)")

    # NOTE: removed JS auto-reload and external st_autorefresh dependency.
    # Use manual controls (Advance Step / Refresh Map) to preserve session_state.

    dummy_ambulances = {
        "AMB-101": {"driver": None, "location": HOSP_COORDS["Kisumu County Referral Hospital"]},
        "AMB-202": {"driver": None, "location": HOSP_COORDS["Jaramogi Oginga Odinga Teaching & Referral Hospital"]}
    }

    # Session state init for map/trip keys
    for key, default in [
        ("active_trip", None),
        ("route_points", None),
        ("route_index", 0),
        ("ambulance_position", None),
        ("driver_name", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Driver login
    with st.expander("🚐 Driver Login"):
        driver_name = st.text_input("Driver Name", key="driver_name_input")
        ambulance_id = st.selectbox("Select Ambulance", list(dummy_ambulances.keys()), key="driver_amb_select")
        if st.button("Login as Driver", key="login_driver"):
            if not driver_name.strip():
                st.error("Enter a driver name first.")
            else:
                st.session_state.driver_name = driver_name.strip()
                origin = dummy_ambulances[ambulance_id]["location"]
                destination = HOSP_COORDS["Jaramogi Oginga Odinga Teaching & Referral Hospital"]
                st.session_state.active_trip = {
                    "ambulance_id": ambulance_id,
                    "origin": origin,
                    "destination": destination,
                }
                # Create interpolated route
                steps = 40
                lat1, lon1 = origin
                lat2, lon2 = destination
                pts = [(lat1 + (lat2 - lat1) * i / steps, lon1 + (lon2 - lon1) * i / steps) for i in range(steps + 1)]
                st.session_state.route_points = pts
                st.session_state.route_index = 0
                st.session_state.ambulance_position = pts[0]
                st.success(f"{st.session_state.driver_name} logged in with {ambulance_id}. Trip started.")

    # Active trip simulation + manual controls
    if st.session_state.active_trip and st.session_state.route_points:
        st.write(f"**Driver:** {st.session_state.driver_name}")
        st.write(f"**Ambulance:** {st.session_state.active_trip['ambulance_id']}")

        # Control buttons
        c1, c2, c3 = st.columns([1,1,1])
        if c1.button("Advance Step", key="advance_step"):
            if st.session_state.route_index < len(st.session_state.route_points) - 1:
                st.session_state.route_index += 1

        if c2.button("Reset Trip", key="reset_trip"):
            st.session_state.active_trip = None
            st.session_state.route_points = None
            st.session_state.route_index = 0
            st.session_state.ambulance_position = None
            st.success("Trip reset.")

        if c3.button("Refresh Map", key="refresh_map"):
            # triggers a rerun which redraws the map but keeps session_state
            st.experimental_rerun()

        # Update current position
        idx = st.session_state.route_index
        current_pos = st.session_state.route_points[idx]
        st.session_state.ambulance_position = current_pos
        destination = st.session_state.active_trip["destination"]

        status = "Arrived" if idx >= len(st.session_state.route_points) - 1 else "En route"
        st.write(f"**Status:** {status}")

        # ETA: assume each step ~0.5 min (adjust if you want)
        minutes_per_step = 0.5
        remaining = len(st.session_state.route_points) - 1 - idx
        eta_minutes = remaining * minutes_per_step
        st.write(f"**ETA:** {eta_minutes:.1f} min")

        # Map drawing with folium
        fmap = folium.Map(location=current_pos, zoom_start=13)

        # Draw route
        folium.PolyLine(st.session_state.route_points, weight=3).add_to(fmap)

        # Start & destination markers
        folium.Marker(
            st.session_state.active_trip["origin"], popup="Origin", icon=folium.Icon(color="green")
        ).add_to(fmap)
        folium.Marker(
            destination, popup="Destination", icon=folium.Icon(color="red")
        ).add_to(fmap)

        # Ambulance marker
        folium.Marker(
            current_pos,
            popup=f"Ambulance {st.session_state.active_trip['ambulance_id']}",
            icon=folium.Icon(color="blue", icon="ambulance", prefix="fa"),
        ).add_to(fmap)

        # Render map
        components.html(fmap._repr_html_(), height=500)

    else:
        st.info("No active trip. Driver, please log in and start a trip above.")

# -----------------------------
# Ambulances Section
# -----------------------------
elif sidebar_choice == "Ambulances":
    st.subheader("🚑 Ambulance Management")

    # Example dummy list of ambulances
    ambulances = [
        {"ID": "AMB-101", "Status": "Available", "Location": "Kisumu County Referral Hospital"},
        {"ID": "AMB-202", "Status": "En Route", "Location": "Jaramogi Oginga Odinga Teaching & Referral Hospital"},
        {"ID": "AMB-303", "Status": "Available", "Location": "Kisumu East Sub-County Hospital"},
    ]

    df_amb = pd.DataFrame(ambulances)
    st.dataframe(df_amb, use_container_width=True)

    # Option to update ambulance status
    selected_amb = st.selectbox("Select Ambulance", df_amb["ID"])
    new_status = st.selectbox("Update Status", ["Available", "En Route", "Under Maintenance"])
    if st.button("Update Status"):
        st.success(f"Status for {selected_amb} updated to {new_status}")


# -----------------------------
# Communications Section
# -----------------------------
elif sidebar_choice == "Communications":
    st.subheader("📞 Communications Center")

    st.text_area("Message to Hospital / Ambulance Team")
    if st.button("Send Message"):
        st.success("Message sent successfully!")

    # Display a log of recent messages (dummy)
    st.write("### Recent Messages")
    messages = [
        {"From": "Dispatch", "To": "AMB-101", "Message": "Pick patient from Nyahera", "Time": "10:45"},
        {"From": "AMB-202", "To": "Referral Center", "Message": "Arrived at JOOTRH", "Time": "10:50"},
    ]
    st.table(pd.DataFrame(messages))


# -----------------------------
# Handover Forms Section
# -----------------------------
elif sidebar_choice == "Handover Forms":
    st.subheader("📋 Patient Handover Form")

    with st.form("handover_form"):
        patient_name = st.text_input("Patient Name")
        condition = st.text_area("Condition Summary")
        meds = st.text_area("Medication Given")
        time_arrival = st.time_input("Time of Arrival", datetime.datetime.now().time())

        submitted = st.form_submit_button("Submit Handover")
        if submitted:
            st.success(f"Handover for {patient_name} recorded at {time_arrival}")


# -----------------------------
# Offline Mode (Simulation)
# -----------------------------
elif sidebar_choice == "Offline":
    st.subheader("📡 Offline Sync (Demo)")

    st.write(
        "This is a simulation of working offline. "
        "In real implementation, data would be cached locally "
        "and synced once internet is restored."
    )

    offline_referrals = [
        {"Patient": "John Doe", "Condition": "Severe asthma", "Hospital": "KCRH", "Status": "Pending Sync"},
        {"Patient": "Jane Smith", "Condition": "Fracture", "Hospital": "JOOTRH", "Status": "Pending Sync"},
    ]
    st.table(pd.DataFrame(offline_referrals))

    if st.button("Sync Now"):
        st.success("✅ All pending records synced successfully.")

