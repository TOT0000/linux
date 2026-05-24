import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from serial.tools import list_ports
import time
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import queue

x_limit = 5
y_limit = 5
z_limit = 5

# setup of the widgets
def create_widgets(window):
    # Style
    style = ttk.Style()
    # style.theme_use('winnative')  # Change the theme as you prefer

    window.title("Real-time UWB Positioning Monitor")
    # window.geometry("1800x1500")
    # window.resizable(False, False)

    # COM port selection
    global com_port_combobox, baud_combobox, com_port_var, baud_var
    com_label = ttk.Label(window, text="Select COM port:", font=('Arial', 20, 'bold'))
    com_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')
    com_port_var = tk.StringVar()
    com_port_combobox = ttk.Combobox(window, textvariable=com_port_var, font=('Arial', 20), state='readonly')
    com_port_combobox.grid(row=0, column=1, padx=10, pady=10, sticky='ew')

    #Refresh button
    refresh_button = ttk.Button(window, text="Refresh", command=populate_com_ports)
    refresh_button.grid(row=0, column=2, padx=10, pady=10, sticky='ew')

    # Populate COM ports
    populate_com_ports()

    # Baud rate selection
    baud_label = ttk.Label(window, text="Select Baud rate:", font=('Arial', 20, 'bold'))
    baud_label.grid(row=0, column=3, padx=10, pady=10, sticky='w')
    baud_var = tk.StringVar()
    baud_combobox = ttk.Combobox(window, textvariable=baud_var, values=["115200", "57600", "28800", "14400", "9600"], font=('Arial', 20), state='readonly')
    baud_combobox.grid(row=0, column=4, padx=10, pady=10, sticky='ew')
    baud_combobox.current(0)

    # Connect button
    global connect_button
    connect_button = ttk.Button(window, text="Connect", command=connect_serial)
    connect_button.grid(row=0, column=5, padx=10, pady=10, sticky='ew')

    # Start animation button
    global start_button
    start_button = ttk.Button(window, text="Start", command=start_animation, state='disabled')
    start_button.grid(row=1, column=5, padx=10, pady=10, sticky='ew')

    # Stop animation button
    global stop_button
    stop_button = ttk.Button(window, text="Stop", command=stop_animation, state='disabled')
    stop_button.grid(row=2, column=5, padx=10, pady=10, sticky='ew')

    # Anchor input
    global anchor_account_entry, enter_anchor_account_button, anchor_coords_entry, enter_anchor_coords_button, reset_button
    anchor_label = ttk.Label(window, text="Enter anchor account:", font=('Arial', 20, 'bold'))
    anchor_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')
    anchor_account_entry = ttk.Entry(window, font=('Arial', 18), state='disabled')
    anchor_account_entry.grid(row=1, column=1, padx=10, pady=10, sticky='ew')
    enter_anchor_account_button = ttk.Button(window, text="Enter", command=enter_anchor_account, state='disabled')
    enter_anchor_account_button.grid(row=1, column=2, padx=10, pady=10, sticky='ew')

    anchor_coords_label = ttk.Label(window, text="Enter anchor coordinates (i,x,y,z):", font=('Arial', 20, 'bold'))
    anchor_coords_label.grid(row=2, column=0, padx=10, pady=10, sticky='w')
    anchor_coords_entry = ttk.Entry(window, font=('Arial', 18), state='disabled')
    anchor_coords_entry.grid(row=2, column=1, padx=10, pady=10, sticky='ew')
    enter_anchor_coords_button = ttk.Button(window, text="Enter", command=enter_anchor_coords, state='disabled')
    enter_anchor_coords_button.grid(row=2, column=2, padx=10, pady=10, sticky='ew')

    #reset button
    reset_button = ttk.Button(window, text="Reset UWB system", command=reset_system, state='disabled')
    reset_button.grid(row=3, column=2 , padx=10, pady=10, sticky='ew')

    # Create a new figure
    global fig, plot1, plot2, plot3, plot4
    fig = plt.figure(figsize=(16, 8), dpi=50)

    # Scatter plot for x and y (largest)
    plot1 = fig.add_subplot(3,6,(1,15))
    # Scatter plot for x and time
    plot2 = fig.add_subplot(3,6,(4,6))
    # Scatter plot for y and time
    plot3 = fig.add_subplot(3,6,(10,12))
    # Scatter plot for z and time
    plot4 = fig.add_subplot(3,6,(16,18))

    getPlotFormat()

    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    fig.tight_layout()

    # Canvas to display the plot
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(row=4, column=0, columnspan=6, rowspan=5, padx=10, pady=10, sticky='nsew')

    # Functional buttons
    global scatter_button, track_button, track_seconds_button, clear_button
    scatter_button = ttk.Button(window, text="Show Real-time Position", command=show_scatter_points, state='disabled')
    scatter_button.grid(row=9, column=0, padx=10, pady=10, sticky='ew')

    track_button = ttk.Button(window, text="Show Track", command=show_track, state='disabled')
    track_button.grid(row=9, column=1, padx=10, pady=10, sticky='ew')

    track_seconds_button = ttk.Button(window, text="Show Track for 10 Seconds", command=show_track_for_10_seconds, state='disabled')
    track_seconds_button.grid(row=9, column=2, padx=10, pady=10, sticky='ew')

    clear_button = ttk.Button(window, text="Clear Path", command=clear_path, state='disabled')
    clear_button.grid(row=9, column=3, padx=10, pady=10, sticky='ew')

    # Make grid cells expandable
    for i in range(3):
        window.grid_rowconfigure(i, weight=1)
    for i in range(6):
        window.grid_columnconfigure(i, weight=1)

# Populate the COM ports in the combobox (done)
def populate_com_ports():
    ports = [port.device for port in list_ports.comports()]
    com_port_combobox['values'] = ports
    if ports:
        com_port_combobox.current(0)

# Connect to the selected serial port (done)
def connect_serial():
    global connected, account_return_status
    if not connected:
        try:
            global ser, stop_flag
            selected_port = com_port_var.get()
            selected_baud = int(baud_var.get())
            ser = serial.Serial(selected_port, selected_baud)
            ser.flushInput()
            print("Connected to", selected_port, "at", "baud rate", selected_baud)
            if ser.is_open:
                connected = True
                account_return_status = True
            #change the state of the connect button and comboboxes
            com_port_combobox['state'] = 'disabled'
            baud_combobox['state'] = 'disabled'
            connect_button['state'] = 'disabled'
            connect_button.grid_forget()
            anchor_account_entry['state'] = 'normal'
            enter_anchor_account_button['state'] = 'normal'
            anchor_account_entry.focus()

            global disconnect_button
            disconnect_button = ttk.Button(root, text="Disconnect", command=disconnect_serial)
            disconnect_button.grid(row=0, column=5, padx=10, pady=10, sticky='ew')
            
            # start a new thread for reading serial data
            # stop_flag = False
            # ser.flushInput()
            # threading.Thread(target=read_serial_data, args=(ser, q), daemon=True).start()
            # root.after(500, start_animation)
        except Exception as e:
            print("Error:", e)
    else:
        print("Already connected to a serial port")
        pass

#disconnect from the serial port (done)
def disconnect_serial():
    # Placeholder function for disconnecting from the serial port
    global connected
    if connected:
        try:
            if ser.is_open:
                ser.close()
                print("Disconnected from the serial port")
                connected = False
                com_port_combobox['state'] = 'normal'
                baud_combobox['state'] = 'normal' 
                disconnect_button.grid_forget()
                connect_button.grid(row=0, column=5, padx=10, pady=10, sticky='ew')
                connect_button['state'] = 'normal'
        except Exception as e:
            print("Error:", e)
    else:
        print("Not connected to any serial port")
        pass

# enter the anchor account (done)
def enter_anchor_account():
    global account_return_status, coords_return_status
    if account_return_status:
        global anchor_account, anchor_account_coords_sent, ser, sending_message
        anchor_account = anchor_account_entry.get()
        if anchor_account and anchor_account.isdigit() and int(anchor_account) > 0 and int(anchor_account) < 10:
            sending_message = anchor_account + ';'
            anchor_account = int(anchor_account)
            print("Anchor count:", anchor_account)
            anchor_account_coords_sent = 0
            anchor_account_entry['state'] = 'disabled'
            anchor_coords_entry['state'] = 'normal'
            enter_anchor_account_button['state'] = 'disabled'
            enter_anchor_coords_button['state'] = 'normal'
            account_return_status = False
            coords_return_status = True
            # after entering the anchor account, the enter key will be bound to enter the anchor coordinates and cursor will be moved to the entry
            root.bind('<Return>', lambda e: enter_anchor_coords())
            anchor_coords_entry.focus()
        else:
            print("Invalid anchor account")
            pass
    else:
        pass

# enter the coordinates of the anchors (need to be modified)
def enter_anchor_coords():
    global coords_return_status, account_return_status
    if coords_return_status:
        global anchor_account_coords_sent, anchor_account, anchor_list, anchor_sent, ready, ser, sending_message, stop_flag
        if anchor_account:
            anchor_coords = anchor_coords_entry.get().strip()  #input format index,x,y,z
            if anchor_coords:
                if not anchor_coords.split(',')[0] in anchor_list:
                    anchor_list.append(anchor_coords.split(',')[0])
                    sending_message += anchor_coords + ';'
                    print("Anchor", anchor_coords.split(',')[0], "coordinates: ", anchor_coords.split(',')[1:])
                    anchor_account_coords_sent += 1
                else:
                    print("Anchor", anchor_coords.split(',')[0], "already exists.")

            if anchor_account_coords_sent == anchor_account:
                ser.flushInput()
                ser.write(sending_message.encode())
                callback_str = ser.readline().decode().strip()
                print("Callback message:", callback_str)
                if callback_str == "Delivery Success":
                    print("UWB system is ready.")
                    anchor_sent = True
                    ready = True
                    stop_flag = False
                    anchor_coords_entry.delete(0, 'end')
                    anchor_coords_entry['state'] = 'disabled'
                    enter_anchor_coords_button['state'] = 'disabled'
                    print("All anchor coordinates sent. Start Positioning.")
                    # print("Sending message:", sending_message)
                    #start a new thread for reading serial data
                    ser.flushInput()
                    root.after(500, threading.Thread(target=read_serial_data, args=(ser, q), daemon=True).start())
                    root.after(10, start_animation)

                    global scatter_button, track_button, track_seconds_button, clear_button, reset_button, start_button, stop_button
                    scatter_button['state'] = 'normal'
                    track_button['state'] = 'normal'
                    track_seconds_button['state'] = 'normal'
                    clear_button['state'] = 'normal'
                    reset_button['state'] = 'normal'
                    start_button['state'] = 'normal'
                    stop_button['state'] = 'normal'
                    account_return_status = False
                    coords_return_status = False
                elif callback_str == "Delivery Fail":
                    print("UWB system is not ready.")
                    anchor_account_coords_sent = 0
                    anchor_list = []
                    anchor_sent = False
                    ready = False
                    stop_flag = True
                    anchor_account_entry['state'] = 'normal'
                    enter_anchor_account_button['state'] = 'normal'
                    anchor_coords_entry['state'] = 'disabled'
                    enter_anchor_coords_button['state'] = 'disabled'
                    anchor_account_entry.delete(0, 'end')
                    anchor_coords_entry.delete(0, 'end')
                    print("Please enter the anchor account again.")
                    root.bind('<Return>', lambda e: enter_anchor_account())
                    anchor_account_entry.focus()
            else:
                anchor_coords_entry.delete(0, 'end')
                print("Please enter the next anchor coordinates.")
                pass
        else:
            print("Please enter the anchor account first.")
            pass
    else:
        pass

# Reset the UWB system (exist bug in the function, need to be modified)
def reset_system():
    global q, dataListBuffer, anchor_list, anchor_sent, ready, ser, stop_flag, anchor_account_coords_sent, account_return_status, coords_return_status
    q.queue.clear()
    dataListBuffer = [[0, 0, 0, 0]]
    anchor_list = []
    anchor_sent = False
    ready = False
    stop_flag = True
    anchor_account_coords_sent = 0
    account_return_status = True
    coords_return_status = False
    root.bind('<Return>', lambda e: enter_anchor_account())
    anchor_account_entry['state'] = 'normal'
    enter_anchor_account_button['state'] = 'normal'
    anchor_coords_entry['state'] = 'disabled'
    enter_anchor_coords_button['state'] = 'disabled'
    anchor_account_entry.delete(0, 'end')
    anchor_coords_entry.delete(0, 'end')
    print("System reset.")
    ser.flushInput()
    ser.write("reset".encode())
    stop_animation()

# Read serial data and put it in the queue (done)
def read_serial_data(ser, q, uwb_callback=None):
    global dataListBuffer, stop_flag
    # ser.timeout = 0.1
    while True:
        if ser.in_waiting:
            try:
                # Read data from the serial port
                data = ser.read(18).decode().strip()
            except Exception as e:
                print("Error reading data:", e)
                ser.flushInput()
                continue
            # Process the data            
            #format: "xx.xx,yy.yy,zz.zz\n"
            print(time.strftime("%Y%m%d_%H%M%S"), data)
            if stop_flag:
                break
            try:
                timestamp = time.time()
                x, y, z = map(float, data.split(','))
                q.put((timestamp, x, y, z))
                dataListBuffer = list(q.queue)[-100:]
               
               # send data to typefly.py
                if uwb_callback is not None:
                    uwb_callback(timestamp, x, y, z)
                
            except Exception as e:
                ser.flushInput()
                print("Error:", e)
        else:
            time.sleep(0.05)

# Animation function (done)
def animate(frame):
    global dataList, mode, ser                         
    
    plot1.clear()                                          
    plot2.clear()                                          
    plot3.clear()                                          
    plot4.clear()                                          
    
    getPlotFormat()
    text = "(" + str(dataListBuffer[-1][1]) + " , " + str(dataListBuffer[-1][2]) + ")"
    plot1.text(dataListBuffer[-1][1], dataListBuffer[-1][2], text, fontsize=14, color='red')
    plot1.scatter(dataListBuffer[-1][1], dataListBuffer[-1][2], )  #just plot the last point
    if mode == "point":
        pass
    elif mode == "track":    
        plot1.plot([d[1] for d in list(q.queue)], [d[2] for d in list(q.queue)], linewidth=3)  #plot trace
    elif mode == "track_10_seconds":
        plot1.plot([d[1] for d in dataListBuffer if time.time() - d[0] <= 10], [d[2] for d in dataListBuffer if time.time() - d[0] <= 10], linewidth=3) #plot last 10 points
    else:
        pass
    if dataListBuffer[-1][1] <= x_limit: plot2.text(99, dataListBuffer[-1][1], str(dataListBuffer[-1][1]), fontsize=14, color='red')
    if dataListBuffer[-1][2] <= y_limit: plot3.text(99, dataListBuffer[-1][2], str(dataListBuffer[-1][2]), fontsize=14, color='red')
    if dataListBuffer[-1][3] <= z_limit: plot4.text(99, dataListBuffer[-1][3], str(dataListBuffer[-1][3]), fontsize=14, color='red')
    plot2.plot([d[1] for d in dataListBuffer], linewidth=3)                                   
    plot3.plot([d[2] for d in dataListBuffer], linewidth=3)                                   
    plot4.plot([d[3] for d in dataListBuffer], linewidth=3)

# Set the format of the plot (done)
def getPlotFormat():
    plot1.set_xlim([0, x_limit])                              
    plot1.set_ylim([0, y_limit])                            
    plot1.grid(True)                                    
    plot1.set_xlabel('X (m)', fontsize=12)
    plot1.set_ylabel('Y (m)', fontsize=12)
    plot1.set_title('X vs Y Position', fontsize=14)                                 
    plot1.xaxis.set_major_locator(plt.MaxNLocator(10))         
    plot1.yaxis.set_major_locator(plt.MaxNLocator(10))

    plot2.get_xaxis().set_visible(True)
    plot2.set_ylim([0, x_limit])
    plot2.grid(True)
    plot2.yaxis.set_major_locator(plt.MaxNLocator(10))
    plot2.set_xlabel('Index', fontsize=12)
    plot2.set_ylabel('X (m)', fontsize=12)
    plot2.set_title('X in Sequence', fontsize=14)
    
    plot3.get_xaxis().set_visible(True)
    plot3.set_ylim([0, y_limit])
    plot3.grid(True)
    plot3.yaxis.set_major_locator(plt.MaxNLocator(10))
    plot3.set_xlabel('Index', fontsize=12)
    plot3.set_ylabel('Y (m)', fontsize=12)
    plot3.set_title('Y in Sequence', fontsize=14)                             
    
    plot4.get_xaxis().set_visible(True)
    plot4.set_ylim([0, z_limit])
    plot4.grid(True)
    plot4.yaxis.set_major_locator(plt.MaxNLocator(10))               
    plot4.set_xlabel('Index', fontsize=12)
    plot4.set_ylabel('Z (m)', fontsize=12)
    plot4.set_title('Z in Sequence', fontsize=14)

# Start the animation (need to be modified)
def start_animation():
    # Placeholder function for starting animation
    global ani
    if 'ani' in globals() and ani:
        ani.event_source.start()
    else:
        ani = animation.FuncAnimation(fig, animate, frames=100, interval=10)
        root.after(100, fig.canvas.draw())

# Stop the animation (done)
def stop_animation():
    if 'ani' in globals() and ani:
        ani.event_source.stop()

# Change the mode to show point (done)
def show_scatter_points():
    # Placeholder function for showing scatter points
    global mode
    mode = "point"

# Change the mode to show track (done)
def show_track():
    # Placeholder function for showing track
    global mode
    mode = "track"

# Change the mode to show track for 10 seconds (done)
def show_track_for_10_seconds():
    # Placeholder function for showing track for 10 seconds
    global mode
    mode = "track_10_seconds"

# clear the path and buffer (done)
def clear_path():
    global dataListBuffer
    dataListBuffer = [[0, 0, 0, 0]]
    q.queue.clear()
    stop_animation()
    root.after(50, start_animation)

# Take a screenshot of the plot
def screenshot():
    global fig
    time_str = time.strftime("%Y%m%d_%H%M%S")
    file_name = time_str + ".png"
    print("Screenshot saved as", file_name)
    fig.savefig(file_name)

# Close the serial port and exit the program (done)
def on_closing():
    try:
        if 'ser' in globals():
            global ser
            if ser.is_open:
                if ready:
                    ser.write("reset".encode())
                root.after(1000, ser.close())
    except Exception as e:
        print("Error:", e)
    root.quit()  # 退出主程式

# Create the main window
# root = tk.Tk()
# create_widgets(root)

# Global variables area
dataListBuffer = [[0, 0, 0, 0]]
connected = False
account_return_status = False
coords_return_status = False
anchor_list = []
anchor_sent = False
ready = False
mode = "point"
q = queue.Queue()



# Start the main loop
if __name__ == "__main__":
    # 這段只會在直接執行 serial_monitor.py 時才跑
    root = tk.Tk()
    create_widgets(root)
    # functional binding area
    root.bind('<Return>', lambda e: enter_anchor_account())
    # root.bind('<Return>', enter_anchor_coords)
    root.bind('<Escape>', lambda e: root.quit())
    root.bind('<Control-s>', lambda e: screenshot())
    root.protocol("WM_DELETE_WINDOW", on_closing)    
    root.mainloop()


  
  



 

