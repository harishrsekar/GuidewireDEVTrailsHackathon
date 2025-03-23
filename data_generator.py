import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_kubernetes_data(n_samples=5000, failure_rate=0.1, time_steps=30):
    """
    Generate synthetic Kubernetes metrics data for training and testing ML models.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    failure_rate : float
        Proportion of samples that should represent failure cases (0-1)
    time_steps : int
        Number of time periods to simulate
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic Kubernetes metrics
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create timestamp series
    start_date = datetime.now() - timedelta(days=time_steps)
    timestamps = [start_date + timedelta(hours=i*24/time_steps) for i in range(time_steps)]
    
    # List to store all data
    all_data = []
    
    # Generate data for each timestamp
    nodes = ['node-' + str(i) for i in range(5)]
    
    for ts in timestamps:
        # Number of samples for this timestamp
        ts_samples = n_samples // time_steps
        
        # Create normal patterns
        for i in range(ts_samples):
            # Assign to random node
            node = np.random.choice(nodes)
            
            # Generate normal metrics
            cpu_usage = np.random.normal(40, 15)
            memory_usage = np.random.normal(50, 10)
            disk_usage = np.random.normal(60, 20)
            network_receive_bytes = np.random.normal(1000000, 500000)
            network_transmit_bytes = np.random.normal(800000, 400000)
            
            # Pod metrics
            pod_count = np.random.randint(5, 20)
            pod_restart_count = np.random.poisson(0.5)
            pod_pending_count = np.random.poisson(0.2)
            
            # Node metrics
            node_condition_ready = 1  # 1=ready, 0=not ready
            node_condition_memory_pressure = 0  # 0=false, 1=true
            node_condition_disk_pressure = 0
            node_condition_pid_pressure = 0
            node_condition_network_unavailable = 0
            
            # Resource quotas
            cpu_request_percentage = np.random.normal(40, 10)
            memory_request_percentage = np.random.normal(50, 15)
            
            # Determine failure status (most are non-failures)
            failure = 0
            
            all_data.append({
                'timestamp': ts,
                'node': node,
                'cpu_usage_percent': max(0, min(100, cpu_usage)),
                'memory_usage_percent': max(0, min(100, memory_usage)),
                'disk_usage_percent': max(0, min(100, disk_usage)),
                'network_receive_bytes': max(0, network_receive_bytes),
                'network_transmit_bytes': max(0, network_transmit_bytes),
                'pod_count': pod_count,
                'pod_restart_count': pod_restart_count,
                'pod_pending_count': pod_pending_count,
                'node_condition_ready': node_condition_ready,
                'node_condition_memory_pressure': node_condition_memory_pressure,
                'node_condition_disk_pressure': node_condition_disk_pressure,
                'node_condition_pid_pressure': node_condition_pid_pressure,
                'node_condition_network_unavailable': node_condition_network_unavailable,
                'cpu_request_percentage': max(0, min(100, cpu_request_percentage)),
                'memory_request_percentage': max(0, min(100, memory_request_percentage)),
                'failure': failure
            })
    
    # Now generate failure cases - we'll replace some of the normal cases
    n_failures = int(n_samples * failure_rate)
    
    # Pick random indices to replace with failure cases
    failure_indices = np.random.choice(range(len(all_data)), size=n_failures, replace=False)
    
    # Different failure patterns
    failure_patterns = [
        # CPU exhaustion
        lambda data: {
            **data,
            'cpu_usage_percent': np.random.uniform(85, 100),
            'pod_restart_count': np.random.randint(3, 10),
            'failure': 1
        },
        # Memory exhaustion
        lambda data: {
            **data,
            'memory_usage_percent': np.random.uniform(90, 100),
            'node_condition_memory_pressure': 1,
            'failure': 1
        },
        # Disk pressure
        lambda data: {
            **data,
            'disk_usage_percent': np.random.uniform(85, 100),
            'node_condition_disk_pressure': 1,
            'failure': 1
        },
        # Network issues
        lambda data: {
            **data,
            'network_receive_bytes': np.random.uniform(0, 100000),
            'network_transmit_bytes': np.random.uniform(0, 50000),
            'node_condition_network_unavailable': 1,
            'failure': 1
        },
        # Node not ready
        lambda data: {
            **data,
            'node_condition_ready': 0,
            'pod_pending_count': np.random.randint(5, 15),
            'failure': 1
        }
    ]
    
    # Apply failure patterns
    for idx in failure_indices:
        pattern = np.random.choice(failure_patterns)
        all_data[idx] = pattern(all_data[idx])
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Ensure the failure column is explicitly converted to integer type
    df['failure'] = df['failure'].astype(int)
    
    # Make sure some failures exist (debug)
    if df['failure'].sum() == 0:
        # Force at least 10% of rows to have failure=1 if none exist
        indices = np.random.choice(df.index, size=int(len(df) * 0.1), replace=False)
        df.loc[indices, 'failure'] = 1
    
    return df
