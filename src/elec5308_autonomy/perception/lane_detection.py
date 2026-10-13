import cv2
import numpy as np
from PIL import Image

from ..types import LaneDetectionResult


class LaneDetector:#create

    def __init__(self):
        print("LaneDetector initialized")

        # Store intermediate masks for debugging / visualization
        self.yellow_mask = None
        self.white_mask = None

        self.yellow_roi = None
        self.white_roi = None

        self.yellow_lines = []
        self.white_lines = []

        self.yellow_fitted_line = None
        self.white_fitted_line = None

        self.roi_points = None

        # Previous driving-area centre line (used only when one
        # lane boundary is temporarily missing).
        self.previous_centerline = None

    # ==================================================
    # MAIN DETECTION FUNCTION
    # ==================================================

    def detect(self, image) -> LaneDetectionResult:

        # ==================================================
        # 1. INPUT VALIDATION
        # ==================================================

        if image is None:
            raise ValueError("Input image cannot be None")

        if isinstance(image, Image.Image):
            image = np.array(image)

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

        elif not isinstance(image, np.ndarray):

            raise TypeError(
                "Input image must be a NumPy array "
                "or PIL Image"
            )

        height, width = image.shape[:2]

        image_center = width / 2

        # ==================================================
        # 2. CONVERT IMAGE TO HSV
        # ==================================================

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        # ==================================================
        # 3. CREATE YELLOW MASK
        # ==================================================

        yellow_mask = cv2.inRange(
            hsv,
            np.array([15, 70, 60]),
            np.array([40, 255, 255])
        )

        # ==================================================
        # 4. CREATE WHITE MASK
        # ==================================================

        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 100]),
            np.array([180, 35 , 255])
        )

        # ==================================================
        # 5. MORPHOLOGY
        #
        # Remove small noise and connect nearby lane fragments
        # ==================================================

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        self.yellow_mask = yellow_mask.copy()
        self.white_mask = white_mask.copy()

        # ==================================================
        # 6. CANNY EDGE DETECTION
        # ==================================================

        yellow_edges = cv2.Canny(
            yellow_mask,
            20,
            80
        )

        white_edges = cv2.Canny(
            white_mask,
            20,
            80
        )

        # ==================================================
        # 7. ROI
        #
        # Trapezoidal road region.
        #
        # The ROI is deliberately narrower at the top
        # and wider at the bottom.
        #
        # This reduces the chance of detecting unrelated
        # road markings or objects outside the driving area.
        # ==================================================

        roi_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        roi_points = np.array(
            [[
                (
                    int(width * 0.26),
                    int(height * 0.55)
                ),
                (
                    int(width * 0.74),
                    int(height * 0.55)
                ),
                (
                    int(width * 0.95),
                    height - 1
                ),
                (
                    int(width * 0.05),
                    height - 1
                )
            ]],
            dtype=np.int32
        )

        self.roi_points = roi_points.copy()

        cv2.fillPoly(
            roi_mask,
            roi_points,
            255
        )

        # ==================================================
        # APPLY ROI
        # ==================================================

        yellow_roi = cv2.bitwise_and(
            yellow_edges,
            roi_mask
        )

        white_roi = cv2.bitwise_and(
            white_edges,
            roi_mask
        )

        self.yellow_roi = yellow_roi.copy()
        self.white_roi = white_roi.copy()

        # ==================================================
        # 8. HOUGH LINE DETECTION
        #
        # Yellow and white are processed separately.
        # ==================================================

        yellow_hough = cv2.HoughLinesP(
            yellow_roi,
            rho=1,
            theta=np.pi / 180,
            threshold=3,
            minLineLength=2,
            maxLineGap=15
        )

        white_hough = cv2.HoughLinesP(
            white_roi,
            rho=1,
            theta=np.pi / 180,
            threshold=6,
            minLineLength=5,
            maxLineGap=12
        )

        # ==================================================
        # 9. FILTER LINE SEGMENTS
        # ==================================================

        def filter_lines(lines, is_white=False):

            valid_lines = []

            if lines is None:
                return valid_lines

            for line in lines:

                x1, y1, x2, y2 = line.reshape(4)

                # ------------------------------------------
                # Make first point the upper point
                # ------------------------------------------

                if y1 > y2:

                    x1, y1, x2, y2 = (
                        x2,
                        y2,
                        x1,
                        y1
                    )

                dx = x2 - x1
                dy = y2 - y1

                # ------------------------------------------
                # Ignore extremely short segments
                # ------------------------------------------

                line_length = np.sqrt(
                    dx ** 2 +
                    dy ** 2
                )

                min_length = 3 if is_white else 5

                if line_length < min_length:
                    continue

                # ------------------------------------------
                # Ignore nearly horizontal lines
                #
                # We calculate dy/dx only when possible.
                #
                # Nearly vertical lines are NOT discarded.
                # =================================================

                if abs(dx) >= 1:

                    slope = dy / dx

                    min_slope = 0.15 if is_white else 0.35

                    if abs(slope) < min_slope:
                        continue

                # ------------------------------------------
                # Keep lines reasonably close to the vehicle
                #
                # At least one endpoint should reach
                # the lower half of the image.
                # ------------------------------------------

                if max(y1, y2) < height * 0.50:
                    continue

                valid_lines.append(
                    [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ]
                )

            return valid_lines

        
        # ==================================================
        # 10. FIT MULTIPLE LINE SEGMENTS
        #
        # Fit all endpoints into one representative line.
        # We use x = a*y + b because lane lines can be
        # close to vertical in the image.
        # ==================================================

        def fit_lane_line(lines, y_top=None, y_bottom=None):

            if len(lines) == 0:
                return None

            points = []

            for line in lines:

                x1, y1, x2, y2 = line

                points.append([x1, y1])
                points.append([x2, y2])

            points = np.array(
                points,
                dtype=np.float32
            )

            if len(points) < 2:
                return None

            y_values = points[:, 1]
            x_values = points[:, 0]

            try:

                a, b = np.polyfit(
                    y_values,
                    x_values,
                    1
                )

            except np.linalg.LinAlgError:

                return None

            # Do not extrapolate a detected line far above the
            # region where it was actually observed.
            # If a common range is supplied, both lane lines are
            # evaluated over the same vertical interval.
            if y_top is None:
                y_top = int(np.min(y_values))

            if y_bottom is None:
                y_bottom = int(np.max(y_values))

            y_top = int(y_top)
            y_bottom = int(y_bottom)

            if y_bottom <= y_top:
                return None

            x_top = int(a * y_top + b)
            x_bottom = int(a * y_bottom + b)

            x_top = int(
                np.clip(
                    x_top,
                    0,
                    width - 1
                )
            )

            x_bottom = int(
                np.clip(
                    x_bottom,
                    0,
                    width - 1
                )
            )

            return [
                x_top,
                y_top,
                x_bottom,
                y_bottom
            ]


        # ==================================================
        # 10. COMBINE ALL VALID LANE SEGMENTS
        #
        # Yellow and white are treated equally.
        # Colour does NOT determine left/right.
        # ==================================================

        yellow_lines = filter_lines(
            yellow_hough
        )

        white_lines = filter_lines(
            white_hough,
            is_white=True
        )

        self.yellow_lines = yellow_lines
        self.white_lines = white_lines


        all_lane_lines = (
            yellow_lines +
            white_lines
        )


        # ==================================================
        # 11. CLASSIFY LINES BY PHYSICAL POSITION
        #
        # Left/right is determined by the bottom x-position.
        # Colour is ignored.
        # ==================================================

        left_lines = []
        right_lines = []

        for line in all_lane_lines:

            x1, y1, x2, y2 = line

            # y2 is the lower endpoint because
            # filter_lines() orders points by y.
            bottom_x = x2

            if bottom_x < image_center:

                left_lines.append(line)

            else:

                right_lines.append(line)


        # ==================================================
        # 12. FIT LEFT AND RIGHT LANE LINES
        # ==================================================

        # Use the vertical overlap of the detected left and
        # right lines. This prevents a short dashed segment from
        # being extrapolated all the way to y=0.40H.
        left_y_min = None
        right_y_min = None

        if left_lines:
            left_y_min = min(line[1] for line in left_lines)

        if right_lines:
            right_y_min = min(line[1] for line in right_lines)

        common_y_top = None
        common_y_bottom = None

        if left_lines and right_lines:
            common_y_top = max(
                left_y_min,
                right_y_min
            )
            # Keep the lower boundary at the bottom of the image.
            # The lower part is the most useful region for steering,
            # so we allow a modest downward extrapolation from the
            # detected lane segments.
            common_y_bottom = height - 1

            # Require a reasonable shared vertical range.
            if common_y_bottom - common_y_top < 10:
                common_y_top = None
                common_y_bottom = None

        left_lane = fit_lane_line(
            left_lines,
            common_y_top,
            common_y_bottom
        )

        right_lane = fit_lane_line(
            right_lines,
            common_y_top,
            common_y_bottom
        )


        # ==================================================
        # OUTPUT INITIALIZATION
        # ==================================================

        driving_area_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        left_lane_output = []
        right_lane_output = []


        # ==================================================
        # 13. DETERMINE ROAD BOUNDARIES
        #
        # The detected line itself is used as the boundary.
        # If only one boundary is visible, estimate the
        # opposite boundary using a fixed road width.
        # ==================================================

        road_width = int(
            width * 0.35
        )

        def previous_center_x(y):
            """Return the previous driving-area centre x at image y."""
            if self.previous_centerline is None:
                return None

            x_top, y_top_prev, x_bottom, y_bottom_prev = (
                self.previous_centerline
            )

            if y_bottom_prev == y_top_prev:
                return float(x_top)

            ratio = (y - y_top_prev) / (y_bottom_prev - y_top_prev)
            return float(
                x_top + ratio * (x_bottom - x_top)
            )


        # --------------------------------------------------
        # BOTH SIDES DETECTED
        # --------------------------------------------------

        if (
            left_lane is not None
            and
            right_lane is not None
        ):

            left_x_top = left_lane[0]
            left_x_bottom = left_lane[2]

            right_x_top = right_lane[0]
            right_x_bottom = right_lane[2]

            left_lane_output = [
                left_lane
            ]

            right_lane_output = [
                right_lane
            ]




        # --------------------------------------------------
        # ONLY LEFT SIDE DETECTED
        # --------------------------------------------------

        elif left_lane is not None:

            left_x_top = left_lane[0]
            left_x_bottom = left_lane[2]

            # If a previous driving-area centre line is available,
            # mirror the detected boundary around that centre line.
            # This preserves the previous lane position instead of
            # assuming a fixed road width.
            previous_top = previous_center_x(left_lane[1])
            previous_bottom = previous_center_x(height - 1)

            if (
                previous_top is not None
                and previous_bottom is not None
            ):
                right_x_top = (
                    2.0 * previous_top - left_x_top
                )
                right_x_bottom = (
                    2.0 * previous_bottom - left_x_bottom
                )
            else:
                right_x_top = left_x_top + road_width
                right_x_bottom = left_x_bottom + road_width

            left_lane_output = [
                left_lane
            ]

    


        # --------------------------------------------------
        # ONLY RIGHT SIDE DETECTED
        # --------------------------------------------------

        elif right_lane is not None:

            right_x_top = right_lane[0]
            right_x_bottom = right_lane[2]

            # If a previous driving-area centre line is available,
            # mirror the detected boundary around that centre line.
            previous_top = previous_center_x(right_lane[1])
            previous_bottom = previous_center_x(height - 1)

            if (
                previous_top is not None
                and previous_bottom is not None
            ):
                left_x_top = (
                    2.0 * previous_top - right_x_top
                )
                left_x_bottom = (
                    2.0 * previous_bottom - right_x_bottom
                )
            else:
                left_x_top = right_x_top - road_width
                left_x_bottom = right_x_bottom - road_width

            right_lane_output = [
                right_lane
            ]

        # --------------------------------------------------
        # NO VALID LINE
        # --------------------------------------------------

        else:

            #print(
            #    "Lane scenario: NO VALID LANE"
            #)

            return LaneDetectionResult(
                left_lane=[],
                right_lane=[],
                driving_area=driving_area_mask
            )
        # ==================================================
        # 14. CLIP VALUES
        # ==================================================

        left_x_top = int(
            np.clip(
                left_x_top,
                0,
                width - 1
            )
        )

        left_x_bottom = int(
            np.clip(
                left_x_bottom,
                0,
                width - 1
            )
        )

        right_x_top = int(
            np.clip(
                right_x_top,
                0,
                width - 1
            )
        )

        right_x_bottom = int(
            np.clip(
                right_x_bottom,
                0,
                width - 1
            )
        )

        # ==================================================
        # 15. GEOMETRY VALIDATION
        # ==================================================

        if (
            left_x_bottom >= right_x_bottom
            or left_x_top >= right_x_top
            or (right_x_bottom - left_x_bottom) < width * 0.15
            or (right_x_bottom - left_x_bottom) > width * 0.85
        ):

            #print(
            #    "Invalid lane geometry"
            #)

            return LaneDetectionResult(
                left_lane=left_lane_output,
                right_lane=right_lane_output,
                driving_area=driving_area_mask
            )

        # ==================================================
        # 16. CREATE DRIVING AREA
        # ==================================================

        # Use the detected overlap for the top, but extend the
        # driving area to the bottom of the camera image.
        y_top = left_lane[1] if left_lane is not None else right_lane[1]
        y_bottom = height - 1

        driving_area_points = np.array(
            [
                [
                    left_x_bottom,
                    y_bottom
                ],
                [
                    left_x_top,
                    y_top
                ],
                [
                    right_x_top,
                    y_top
                ],
                [
                    right_x_bottom,
                    y_bottom
                ]
            ],
            dtype=np.int32
        )

        cv2.fillPoly(
            driving_area_mask,
            [driving_area_points],
            255
        )

        # Save the current driving-area centre line. It is used only
        # as a short-term fallback if one boundary disappears in the
        # next frame.
        self.previous_centerline = (
            (left_x_top + right_x_top) / 2.0,
            y_top,
            (left_x_bottom + right_x_bottom) / 2.0,
            y_bottom
        )

        # ==================================================
        # DEBUG INFORMATION
        # ==================================================

       # print(
       #     "Lane scenario:",
       #     scenario
        #)

       # print(
       #     "Yellow segments:",
       #     len(yellow_lines)
        #)

        #print(
        #    "White segments:",
        #    len(white_lines)
        #)

        #print(
        #    "Fitted left:",
        #    left_lane
        #)

        #print(
        #    "Fitted right:",
        #    right_lane
        #)

        #print(
        #    "Driving area:",
        #    driving_area_points.tolist()
        #)

        # ==================================================
        # RETURN RESULT
        # ==================================================

        #print(
        #    "\n========== LANE DETECTION OUTPUT =========="
        #)

        #print(
        #    "left_lane:",
        #    left_lane_output
        #)

        #print(
        #    "right_lane:",
        #    right_lane_output
        #)

        #print(
        #    "driving_area shape:",
        #    driving_area_mask.shape
        #)

        #print(
        #    "===========================================\n"
        #)

        return LaneDetectionResult(
            left_lane=left_lane_output,
            right_lane=right_lane_output,
            driving_area=driving_area_mask
        )