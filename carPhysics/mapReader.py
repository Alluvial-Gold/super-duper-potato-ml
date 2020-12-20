
import lxml
import lxml.etree

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def bezierEval(points):
    N = 50
    t = np.linspace(0, 1, N)
    if points.shape[1] == 4:
        #it is a cubic bezier curve
        c1 = 3*(points[:, 1] - points[:, 0])
        c2 = 3*(points[:, 0] - 2*points[:, 1] + points[:, 2])
        c3 = -points[:, 0] + 3*(points[:, 1] - points[:, 2]) + points[:, 3]
        
        
        c1.shape = (2, 1)
        c2.shape = (2, 1)
        c3.shape = (2, 1)
        
        print(c1)
        print(c2)
        print(c3)
        
        Q = np.power(t, 3)*c3 + np.power(t, 2)*c2 + t*c1+ np.tile(points[:, 0], [N, 1]).transpose()
        return Q
    else:
        raise Exception('Only cubic beziers are supported')
        

def read_svg_map(filename, debug_plot_map=False):


    tree = lxml.etree.parse(filename)


    namespaces = {"svg": "http://www.w3.org/2000/svg",
                  "inkscape": "http://www.inkscape.org/namespaces/inkscape"}

    # Get the walls
    wall_layer_path = r"//svg:g[@inkscape:label='walls']"
    wall_layer = tree.xpath(wall_layer_path, namespaces=namespaces)[0]
    wall_path_path = "./svg:path"
    wall_path_elements = wall_layer.xpath(wall_path_path, namespaces=namespaces)

    def convert_pos(s):
        return np.array( [float(p) for p in s.split(",")] )


    all_wall_points = []

    if debug_plot_map:
        fig, ax = plt.subplots()


    for wall_path_elem in wall_path_elements:
        data_text = wall_path_elem.get("d")

        data_parts = data_text.split()

        # Reverse it to allow efficient pop
        data_parts = data_parts[::-1]

        state = "INIT"

        part = data_parts.pop()
        last_pos = np.full((2, 1), np.nan)

        points = []

        while state != "END":

            if state == "INIT":
                # First part should be a "M"

                marker = part
                assert(marker.upper()=="M")

                # Starting coordinate
                part = data_parts.pop()
                pos = convert_pos(part)
                points.append(pos)
                if( marker=="m" ):
                    # Relative location
                    pos_start = pos
                else:
                    # Absolute location
                    pos_start = np.array([0,0])

                next_state = "STARTED"

            elif state == "STARTED":

                # "C" marker
                assert(part.upper()=="C")

                # Control coordinate 1
                part = data_parts.pop()
                pos = convert_pos(part) + pos_start
                points.append(pos)

                # Control coordinate 2
                part = data_parts.pop()
                pos = convert_pos(part) + pos_start
                points.append(pos)

                # End coordinate
                part = data_parts.pop()
                pos = convert_pos(part) + pos_start
                points.append(pos)

                last_pos = points[-1]
                next_state = "END"



            # Prepare for next
            if next_state != "END":
                part = data_parts.pop()
            state = next_state


        points = np.vstack(points)
        # Flip y axis
        points[:, 1] *= -1

        print(points)

        curveEval = bezierEval(points.T).T
        all_wall_points.append(curveEval)

        if debug_plot_map:
            c = ax.plot(curveEval[:, 0], curveEval[:, 1])[0].get_color()
            ax.plot(points[0, 0], points[0, 1], 'o', c=c, label="Start")
            ax.plot(points[1, 0], points[1, 1], 'o', c=c, label="control_1")
            ax.plot(points[2, 0], points[2, 1], 'o', c=c, label="control_2")
            ax.plot(points[3, 0], points[3, 1], 'o', c=c, label="End")

       # ax.legend()


    # Get gates
    gates_layer_path = r"//svg:g[@inkscape:label='gates']"
    gates_path = "./svg:rect"
    gates_layer = tree.xpath(gates_layer_path, namespaces=namespaces)[0]
    gates_elements = gates_layer.xpath(gates_path, namespaces=namespaces)

    all_rect_coords = []
    for gate_elem in gates_elements:
        keys = ["width", "height", "x", "y"]
        rect_data = {}
        for key in keys:
            rect_data[key] = float( gate_elem.get(key) )
        rect_data["number"] = int( gate_elem.get("id") )

        # Flip y
        rect_data["y"] *= -1
        rect_data["y"] -= rect_data["height"]
        
        all_rect_coords.append(rect_data)

        if debug_plot_map:
            patch = mpl.patches.Rectangle((rect_data["x"], rect_data["y"]), rect_data["width"], rect_data["height"])
            ax.add_patch(patch)

            center_x = rect_data["x"] + rect_data["width"]/2
            center_y = rect_data["y"] + rect_data["height"]/2
            ax.text(center_x, center_y, "Gate %d" % rect_data["number"], 
                    horizontalalignment="center", verticalalignment="center")

    # Get meta/other
    meta_layer_path = r"//svg:g[@inkscape:label='meta']"
    start_path = r"./svg:ellipse[@id='start']"
    meta_layer = tree.xpath(meta_layer_path, namespaces=namespaces)[0]

    start_elem = meta_layer.xpath(start_path, namespaces=namespaces)[0]
    start_coordinate = (float( start_elem.get("cx") ), -float( start_elem.get("cy") ))  # Note y flipped here

    if debug_plot_map:
        ax.plot(start_coordinate[0], start_coordinate[1], "ko", label="Race Start")



    if debug_plot_map:
        ax.axis("equal")
        plt.show()

    return (all_wall_points, all_rect_coords, start_coordinate)

if __name__ == "__main__":
    filename = "test.svg"

    all_wall_points, all_rect_coords, start_pos = read_svg_map(filename, True)


        



    


