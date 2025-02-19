import math
import datetime
import pandas


def main1():
    alphas = [30, 45, 60, 75]
    alts = [200, 300, 400, 500, 600]
    earth = 6371
    combs = [(alpha, alt) for alpha in alphas for alt in alts]
    for a, h in combs:
        rad_a = math.radians(a)
        sin_b = (-earth * math.sin(rad_a) * math.cos(rad_a) + math.cos(rad_a) *
                 math.sqrt(earth * earth * math.sin(rad_a) * math.sin(rad_a) + 2 * earth * h + h * h))/(earth + h)
        rad_b = math.asin(sin_b)
        b = math.degrees(rad_b)
        s = f"alpha = {a}, alt = {h}, sin_b = {sin_b}, rad_b = {rad_b}, b = {b}"
        print(s)


def get_angle(elevation_angle, altitude):
    rad_a = math.radians(elevation_angle)
    cos_a = math.cos(rad_a)
    sin_a = math.sin(rad_a)
    earth = 6371
    sin_b = (-earth * sin_a * cos_a + cos_a *
             math.sqrt(earth * earth * sin_a * sin_a + 2 * earth * altitude + altitude * altitude)) / (earth + altitude)
    rad_b = math.asin(sin_b)
    return rad_b

def get_endpoint(lat1, lon1, bearing, distance):
    # Earth's radius in kilometers
    R = 6371

    # Convert bearing from degrees to radians
    brng = math.radians(bearing)

    # Convert lat1 and lon1 to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)

    # Calculate new latitude and longitude
    lat2 = math.asin(math.sin(lat1_rad) * math.cos(distance / R) + math.cos(lat1_rad) * math.sin(distance / R) * math.cos(brng))
    lon2 = lon1_rad + math.atan2(math.sin(brng) * math.sin(distance / R) * math.cos(lat1_rad), math.cos(distance / R) - math.sin(lat1_rad) * math.sin(lat2))

    # Convert lat2 and lon2 back to degrees
    lat2_deg = math.degrees(lat2)
    lon2_deg = math.degrees(lon2)

    return lat2_deg, lon2_deg

# Example usage
# starting_lat = 28.455556
# starting_lon = -80.527778
# bearing_degrees = 317.662819
# distance_nautical_miles = 130.224835

# end_lat, end_lon = get_endpoint(starting_lat, starting_lon, bearing_degrees, distance_nautical_miles)
# print(f"Destination coordinates: Latitude {end_lat:.6f}, Longitude {end_lon:.6f}")

def main2():
    starting_lat = 59.63
    starting_lon = 17.25
    azimuth = 158.66
    elevation_angle = 35.72
    altitude = 350
    distance = get_angle(elevation_angle, altitude) * 6371
    end_point = get_endpoint(starting_lat, starting_lon, azimuth, distance)
    print(end_point)

if __name__ == "__main__":
    # main2()
    s1 = {"a": 1, "b": 2}
    print(math.sin(0.1))
