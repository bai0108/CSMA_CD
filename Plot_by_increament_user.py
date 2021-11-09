# CSMA/CD Algorithm
import random
import math
import collections
import matplotlib.pyplot as plt

maxSimulationTime = 100
total_num = 0


class Node:
    def __init__(self, location, A, max_collision=15):
        self.queue = collections.deque(self.generate_queue(A))
        self.location = location  # Defined as a multiple of D
        self.collisions = 0
        self.wait_collisions = 0
        # problem 3 max collisions is 15 while original one is 10
        self.MAX_COLLISIONS = max_collision

    def collision_occured(self, R):
        self.collisions += 1
        if self.collisions > self.MAX_COLLISIONS:
            # Drop packet and reset collisions
            return self.pop_packet()

        # ---- problem 3 ---- #
        if 10 < self.collisions < self.MAX_COLLISIONS:
            self.collisions = 10

        # Add the exponential backoff time to waiting time
        backoff_time = self.queue[0] + self.exponential_backoff_time(R, self.collisions)

        for i in range(len(self.queue)):
            if backoff_time >= self.queue[i]:
                self.queue[i] = backoff_time
            else:
                break

    def successful_transmission(self):
        self.collisions = 0
        self.wait_collisions = 0

    def generate_queue(self, A):
        packets = []
        arrival_time_sum = 0

        while arrival_time_sum <= maxSimulationTime:
            arrival_time_sum += get_exponential_random_variable(A)
            packets.append(arrival_time_sum)
        return sorted(packets)

    def exponential_backoff_time(self, R, general_collisions):
        # Original
        if self.MAX_COLLISIONS == 10:
            rand_num = random.random() * (pow(2, general_collisions) - 1)
        # ---- problem 3 ---- #
        if self.MAX_COLLISIONS == 15:
            rand_num = random.randint(0, pow(2, general_collisions) - 1)
        return rand_num * 512 / float(R)  # 512 bit-times

    def pop_packet(self):
        self.queue.popleft()
        self.collisions = 0
        self.wait_collisions = 0

    def non_persistent_bus_busy(self, R):
        self.wait_collisions += 1
        if self.wait_collisions > self.MAX_COLLISIONS:
            # Drop packet and reset collisions
            return self.pop_packet()

        # Add the exponential backoff time to waiting time
        backoff_time = self.queue[0] + self.exponential_backoff_time(R, self.wait_collisions)

        for i in range(len(self.queue)):
            if backoff_time >= self.queue[i]:
                self.queue[i] = backoff_time
            else:
                break


def get_exponential_random_variable(param):
    # Get random value between 0 (exclusive) and 1 (inclusive)
    uniform_random_value = 1 - random.uniform(0, 1)
    exponential_random_value = (-math.log(1 - uniform_random_value) / float(param))

    return exponential_random_value


def build_nodes(N, A, D, max_collision):
    nodes = []
    for i in range(0, N):
        nodes.append(Node(i * D, A, max_collision))
    return nodes


def csma_cd(N, A, R, L, D, S, is_persistent, max_collision):
    curr_time = 0
    transmitted_packets = 0
    successfully_transmitted_packets = 0
    nodes = build_nodes(N, A, D, max_collision)

    while True:

        # Step 1: Pick the smallest time out of all the nodes
        min_node = Node(None, A, max_collision)  # Some random temporary node
        min_node.queue = [float("infinity")]
        for node in nodes:
            if len(node.queue) > 0:
                min_node = min_node if min_node.queue[0] < node.queue[0] else node

        if min_node.location is None:  # Terminate if no more packets to be delivered
            break

        curr_time = min_node.queue[0]
        transmitted_packets += 1

        # Step 2: Check if collision will happen
        # Check if all other nodes except the min node will collide
        collision_occurred_once = False
        for node in nodes:
            if node.location != min_node.location and len(node.queue) > 0:
                delta_location = abs(min_node.location - node.location)
                t_prop = delta_location / float(S)
                t_trans = L / float(R)

                # Check collision
                will_collide = True if node.queue[0] <= (curr_time + t_prop) else False

                # Sense bus busy
                if (curr_time + t_prop) < node.queue[0] < (curr_time + t_prop + t_trans):
                    if is_persistent is True:
                        for i in range(len(node.queue)):
                            if (curr_time + t_prop) < node.queue[i] < (curr_time + t_prop + t_trans):
                                node.queue[i] = (curr_time + t_prop + t_trans)
                            else:
                                break
                    else:
                        node.non_persistent_bus_busy(R)

                if will_collide:
                    collision_occurred_once = True
                    transmitted_packets += 1
                    node.collision_occured(R)

        # Step 3: If a collision occurred then retry
        # otherwise update all nodes latest packet arrival times and proceed to the next packet
        if collision_occurred_once is not True:  # If no collision happened
            successfully_transmitted_packets += 1
            min_node.pop_packet()
        else:  # If a collision occurred
            min_node.collision_occured(R)
    eff = successfully_transmitted_packets / float(transmitted_packets)
    throput = (L * successfully_transmitted_packets) / float(curr_time + (L / R)) * 1e-6
    print("Efficiency", eff)
    print("Throughput", throput, "Mbps")
    print("")
    return eff, throput


# Run Algorithm
# N = The number of nodes/computers connected to the LAN
# A = Average packet arrival rate (packets per second)
# R = The speed of the LAN/channel/bus (in bps)
# L = Packet length (in bits)
# D = Distance between adjacent nodes on the bus/channel
# S = Propagation speed (meters/sec)

if __name__ == '__main__':
    D = 10
    C = 3 * 1e8  # speed of light
    S = (2 / float(3)) * C
    A = 10
    user_num = [i for i in range(20, 101, 20)]
    # img 1
    Efficiency_or_is = []
    Efficiency_or_non = []
    Efficiency_re_is = []
    Efficiency_re_non = []
    # img 2
    Throughput_or_is = []
    Throughput_or_non = []
    Throughput_re_is = []
    Throughput_re_non = []
    # Show the efficiency and throughput of the LAN (in Mbps) (CSMA/CD Persistent)
    for N in range(20, 101, 20):
        # for A in [7, 10, 20]:
        R = 1e6
        L = 1500
        print("Or_1-Persistent: ", "Nodes: ", N, "Avg Packet: ", A)
        eff_or_is, thro_or_is = csma_cd(N, A, R, L, D, S, True, max_collision=10)
        print("Re_1-Persistent: ", "Nodes: ", N, "Avg Packet: ", A)
        eff_re_is, thro_re_is = csma_cd(N, A, R, L, D, S, True, max_collision=15)
        print("Or_non-Persistent: ", "Nodes: ", N, "Avg Packet: ", A)
        eff_or_non, thro_or_non = csma_cd(N, A, R, L, D, S, False, max_collision=10)
        print("Re_non-Persistent: ", "Nodes: ", N, "Avg Packet: ", A)
        eff_re_non, thro_re_non = csma_cd(N, A, R, L, D, S, False, max_collision=15)
        Efficiency_or_is.append(eff_or_is)
        Efficiency_or_non.append(eff_or_non)
        Efficiency_re_is.append(eff_re_is)
        Efficiency_re_non.append(eff_re_non)

        Throughput_or_is.append(thro_or_is)
        Throughput_or_non.append(thro_or_non)
        Throughput_re_is.append(thro_re_is)
        Throughput_re_non.append(thro_re_non)
    # write results to file
    with open('res_file.txt', 'w') as f:
        f.write("%s\n" % Efficiency_or_is)
        f.write("%s\n" % Efficiency_or_non)
        f.write("%s\n" % Efficiency_re_is)
        f.write("%s\n" % Efficiency_re_non)
        f.close()

    with open('res_file.txt', 'a') as f:
        f.write("%s\n" % Throughput_or_is)
        f.write("%s\n" % Throughput_or_non)
        f.write("%s\n" % Throughput_re_is)
        f.write("%s\n" % Throughput_re_non)
        f.close()
    # plot
    plt.figure(1)
    plt.title("Img 1")
    plt.xlabel("Number of user")
    plt.ylabel("Efficiency")
    plt.plot(user_num, Efficiency_or_is, 'g', label='Original_1_Persistent')
    plt.plot(user_num, Efficiency_or_non, 'b', label='Original_non_Persistent')
    plt.plot(user_num, Efficiency_re_is, 'm', label='Revise_1_Persistent')
    plt.plot(user_num, Efficiency_re_non, 'c', label='Revise_non_Persistent')
    plt.legend()

    plt.figure(2)
    plt.title("Img 2")
    plt.xlabel("Number of user")
    plt.ylabel("Throughput")
    plt.plot(user_num, Throughput_or_is, 'g', label='Original_1_Persistent')
    plt.plot(user_num, Throughput_or_non, 'b', label='Original_non_Persistent')
    plt.plot(user_num, Throughput_re_is, 'm', label='Revise_1_Persistent')
    plt.plot(user_num, Throughput_re_non, 'c', label='Revise_non_Persistent')
    plt.legend()
    plt.show()
    # Show the efficiency and throughput of the LAN (in Mbps) (CSMA/CD Non-persistent)
    # for N in range(20, 101, 20):
    #     # for A in [7, 10, 20]:
    #     R = 1 * pow(10, 6)
    #     L = 1500
    #     print("Non persistent", "Nodes: ", N, "Avg Packet: ", A)
    #     csma_cd(N, A, R, L, D, S, False)
