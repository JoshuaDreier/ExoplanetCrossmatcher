from typing import Tuple, List, Dict
import numpy as np
from dataclasses import dataclass

"""
Crossmatching module for astronomical catalog matching.

Combines ID-based and coordinate-based crossmatching algorithms.
First attempts to match by ID, then performs coordinate-based matching
on unmatched sources.
"""




@dataclass
class CrossmatchResult:
    """Store results from crossmatching operation."""
    matched_pairs: List[Tuple[int, int]]  # (source_idx, catalog_idx)
    unmatched_sources: List[int]
    unmatched_catalog: List[int]
    match_distances: List[float]  # distances for coordinate matches


class Crossmatcher:
    """
    Combines ID-based and coordinate-based crossmatching.
    
    First matches objects by ID, then performs coordinate-based
    matching on remaining unmatched sources.
    """
    
    def __init__(self, coord_tolerance: float = 1.0):
        """
        Initialize crossmatcher.
        
        Args:
            coord_tolerance: Maximum distance (arcsec) for coordinate match.
        """
        self.coord_tolerance = coord_tolerance
        self.id_matches = []
        self.coord_matches = []
    
    def crossmatch(
        self,
        source_ids: np.ndarray,
        source_coords: np.ndarray,
        catalog_ids: np.ndarray,
        catalog_coords: np.ndarray
    ) -> CrossmatchResult:
        """
        Perform combined ID and coordinate crossmatching.
        
        Args:
            source_ids: Array of source IDs
            source_coords: Array of (RA, Dec) coordinates for sources
            catalog_ids: Array of catalog IDs
            catalog_coords: Array of (RA, Dec) coordinates for catalog
            
        Returns:
            CrossmatchResult with matched pairs and unmatched indices.
        """
        # Step 1: ID-based matching
        id_matched_sources, id_matched_catalog = self._match_by_id(
            source_ids, catalog_ids
        )
        
        # Step 2: Identify unmatched sources and catalog entries
        all_sources = set(range(len(source_ids)))
        all_catalog = set(range(len(catalog_ids)))
        unmatched_sources = list(all_sources - set(id_matched_sources))
        unmatched_catalog = list(all_catalog - set(id_matched_catalog))
        
        # Step 3: Coordinate-based matching on unmatched sources
        coord_pairs, distances = self._match_by_coordinates(
            source_coords[unmatched_sources],
            catalog_coords[unmatched_catalog],
            unmatched_sources,
            unmatched_catalog
        )
        
        # Combine results
        matched_pairs = list(zip(id_matched_sources, id_matched_catalog)) + coord_pairs
        coord_distances = [np.inf] * len(id_matched_sources) + distances
        
        final_unmatched_sources = [
            s for s in unmatched_sources
            if s not in [p[0] for p in coord_pairs]
        ]
        final_unmatched_catalog = [
            c for c in unmatched_catalog
            if c not in [p[1] for p in coord_pairs]
        ]
        
        return CrossmatchResult(
            matched_pairs=matched_pairs,
            unmatched_sources=final_unmatched_sources,
            unmatched_catalog=final_unmatched_catalog,
            match_distances=coord_distances
        )
    
    def _match_by_id(
        self,
        source_ids: np.ndarray,
        catalog_ids: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Match sources to catalog by ID."""
        matched_sources = []
        matched_catalog = []
        
        for src_idx, src_id in enumerate(source_ids):
            cat_matches = np.where(catalog_ids == src_id)[0]
            if len(cat_matches) > 0:
                matched_sources.append(src_idx)
                matched_catalog.append(cat_matches[0])
        
        self.id_matches = list(zip(matched_sources, matched_catalog))
        return matched_sources, matched_catalog
    
    def _match_by_coordinates(
        self,
        source_coords: np.ndarray,
        catalog_coords: np.ndarray,
        source_indices: List[int],
        catalog_indices: List[int]
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Match sources to catalog by coordinates."""
        matched_pairs = []
        distances = []
        matched_catalog = set()
        
        for src_idx_local, src_idx_global in enumerate(source_indices):
            distances_to_catalog = self._angular_distance(
                source_coords[src_idx_local],
                catalog_coords
            )
            
            closest_cat_local = np.argmin(distances_to_catalog)
            closest_distance = distances_to_catalog[closest_cat_local]
            closest_cat_global = catalog_indices[closest_cat_local]
            
            if (closest_distance <= self.coord_tolerance and
                closest_cat_global not in matched_catalog):
                matched_pairs.append((src_idx_global, closest_cat_global))
                distances.append(closest_distance)
                matched_catalog.add(closest_cat_global)
        
        self.coord_matches = matched_pairs
        return matched_pairs, distances
    
    @staticmethod
    def _angular_distance(coord1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """
        Calculate angular distance in arcseconds using haversine formula.
        
        Args:
            coord1: Single (RA, Dec) coordinate in degrees
            coords2: Array of (RA, Dec) coordinates in degrees
            
        Returns:
            Array of distances in arcseconds
        """
        ra1, dec1 = np.radians(coord1)
        ra2, dec2 = np.radians(coords2.T)
        
        dlon = ra2 - ra1
        dlat = dec2 - dec1
        
        a = np.sin(dlat/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return np.degrees(c) * 3600  # Convert to arcseconds


def create_crossmatcher(coord_tolerance: float = 1.0) -> Crossmatcher:
    """Factory function to create a new Crossmatcher instance."""
    return Crossmatcher(coord_tolerance=coord_tolerance)