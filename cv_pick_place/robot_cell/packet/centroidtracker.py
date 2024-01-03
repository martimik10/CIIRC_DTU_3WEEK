import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker:
    """
    Class for tracking packets between frames.
    """

    def __init__(self, maxDisappeared: int = 20) -> None:
        """
        CentroidTracker object constructor.

        Args:
            maxDisappeared (int): Maximum number of frames before deregister.
        """

        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, item: np.ndarray):
        """
        Registers input item.

        Args:
            item (np.ndarray): Numpy array with items to be registered.
        """

        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = item
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID: str):
        """
        Deregisters object based on ID.

        Args:
            objectID (str): Key to deregister items in the objects dict.
        """

        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects: list) -> OrderedDict:
        """
        Updates the currently tracked centroids.

        Args:
            rects (list): List containing boundig box points to be tracked.

        Returns:
            OrderedDict: Ordered dictionary with tracked centroids.
        """

        # is box empty
        if len(rects) == 0:
            # loop overobjects and mark them as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # deregister if maximum number of consecutive frames where missing
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects
        # array of input centroids at current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # inputCentroids = centroid
        # # loop over the bounding box rectangles
        for i in range(0, len(rects)):
            # # use the bounding box coordinates to derive the centroid
            inputCentroids[i] = rects[i][1]
        # if not tracking any objects take input centroids, register them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        # else, while currently tracking objects try match the input centroids to existing centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        # return the set of trackable objects
        return self.objects

    def update_detected(self, detected: list) -> OrderedDict:
        """
        Updates the currently tracked detections.

        Args:
            detected (list): List containing detections to be tracked.

        Returns:
            OrderedDict: Ordered dictionary with tracked detections.
        """

        if len(detected) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        inputData = np.zeros((len(detected), 4))

        for i in range(0, len(detected)):
            inputData[i, :2] = detected[i][1]
            inputData[i, 2] = detected[i][2]
            inputData[i, 3] = detected[i][3]
        if len(self.objects) == 0:
            for i in range(0, len(inputData)):
                self.register(inputData[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [
                element[:2] for i, element in enumerate(list(self.objects.values()))
            ]
            D = dist.cdist(np.array(objectCentroids), inputData[:, :2])
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputData[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputData[col])
        return self.objects
